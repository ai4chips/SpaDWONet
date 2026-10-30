import sys, os, time
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp, numpy as np, optax, equinox as eqx


class Tee:
    def __init__(self, fname, mode='w'):
        self.file = open(fname, mode)
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from model_dwno_2d_jax import DWNO2d, count_params


ntrain, ntest, batch_size = 1000, 100, 25
learning_rate, epochs, weight_decay = 0.001, 1000, 1e-4
width, level, P, s = 64, 4, 25, 64

data_dir = os.path.join(_this_dir, '..', '..', 'data', '2d')
save_dir = os.path.join(_this_dir, 'results')


def rel_l2(pred, true):
    return jnp.linalg.norm((pred - true).ravel()) / jnp.linalg.norm(true.ravel())


def train(num_param_levels=1, num_wavelet_layers=6, P_val=25, suffix='baseline', data_subdir=""):
    save_path = os.path.join(save_dir, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    _data_dir = os.path.join(data_dir, data_subdir) if data_subdir else data_dir
    print(f'JAX backend: {"GPU" if jax.devices()[0].platform == "gpu" else "CPU"}')
    print(f'Model: DWNO2d JAX, level={level}, width={width}, P={P_val}')
    print(f'Param levels: {num_param_levels}/{level}, Wavelet layers: {num_wavelet_layers}')
    print(f'Data: {ntrain} train, {ntest} test, grid {s}x{s}, batch={batch_size}')

    
    print('\nLoading data...')
    f_train_raw = jnp.array(np.load(os.path.join(_data_dir, 'f_train.npy')), dtype=jnp.float32)
    u_train_raw = jnp.array(np.load(os.path.join(_data_dir, 'u_train.npy')), dtype=jnp.float32)
    f_test_raw = jnp.array(np.load(os.path.join(_data_dir, 'f_test.npy')), dtype=jnp.float32)
    u_test_raw = jnp.array(np.load(os.path.join(_data_dir, 'u_test.npy')), dtype=jnp.float32)

    f_max = float(f_train_raw.max())
    f_train = f_train_raw[:ntrain] / f_max
    u_train = u_train_raw[:ntrain]
    f_test = f_test_raw[:ntest] / f_max
    u_test = u_test_raw[:ntest]

    f_train = f_train[..., None]  
    u_train = u_train[..., None]
    f_test = f_test[..., None]
    u_test = u_test[..., None]

    print(f'  f_train: {f_train.shape}, u_train: {u_train.shape}')
    print(f'  f_max: {f_max:.2f}  u range: [{float(u_train.min()):.4f}, {float(u_train.max()):.4f}]')

    
    print('\nCreating model...')
    key = jax.random.PRNGKey(0)
    model = DWNO2d(width=width, level=level, s=s, P=P_val,
                   num_param_levels=num_param_levels,
                   num_wavelet_layers=num_wavelet_layers, key=key)
    n_params = count_params(model)
    print(f'  Parameters: {n_params:,}')

    
    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(optax.clip_by_global_norm(1.0),
                        optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(src) - tgt)**2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss_val

    @eqx.filter_jit
    def eval_batch(model, src):
        return model(src)

    def eval_all(model, src_all, tgt_all, eb=25):
        preds = []
        for start in range(0, src_all.shape[0], eb):
            end = min(start + eb, src_all.shape[0])
            preds.append(np.array(eval_batch(model, src_all[start:end])))
        pred_all = jnp.concatenate(preds, axis=0)
        d = pred_all - tgt_all
        return jnp.mean(d**2), rel_l2(pred_all, tgt_all), pred_all, \
               jnp.mean(jnp.abs(d)), jnp.max(jnp.abs(d))

    
    train_loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_time = 0.0
    log_every = max(1, epochs // 100)

    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain, batch_size, replace=False)
        model, opt_state, loss_val = train_step(model, opt_state, f_train[idx], u_train[idx])
        if ep % log_every == 0 or ep == epochs - 1:
            _, test_rel, _, test_mae, test_max = eval_all(model, f_test, u_test)
            trv, tmv, txv = test_rel.item(), test_mae.item(), test_max.item()
        else:
            trv = test_rel_l2_log[max(0, ep-1)] if ep > 0 else 1.0
            tmv = test_mae_log[max(0, ep-1)] if ep > 0 else 0.0
            txv = test_max_log[max(0, ep-1)] if ep > 0 else 0.0
        t2 = time.perf_counter()
        total_time += (t2 - t1)
        train_loss_log[ep] = loss_val.item()
        test_rel_l2_log[ep] = trv; test_mae_log[ep] = tmv; test_max_log[ep] = txv
        if ep % log_every == 0:
            print(f'Ep {ep:5d} | T {t2-t1:.1f}s | MSE {loss_val.item():.6f} | '
                  f'Rel.L2 {trv:.6f} MAE {tmv:.4f} MaxAbs {txv:.4f}')

    print(f'\n{"="*60}')
    print(f'Training complete. Total time: {total_time:.2f}s')
    print(f'Final Test Rel.L2: {test_rel_l2_log[-1]:.6f} ({100*test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.4f}  MaxAbs: {test_max_log[-1]:.4f}')
    print(f'{"="*60}\n')

    for d in jax.devices():
        if d.platform == 'gpu':
            mem = d.memory_stats()
            if mem:
                peak = mem.get('peak_bytes_in_use', 0) / 1e9
                print(f'GPU Memory Used (peak): {peak:.2f} GB')

    _, _, pred_all, tmae, tmax = eval_all(model, f_test, u_test)
    pred_np = np.array(pred_all)[..., 0]
    u_test_np = np.array(u_test)[..., 0]
    sample_errors = np.zeros(ntest, dtype=np.float64)
    for i in range(ntest):
        sample_errors[i] = rel_l2(jnp.array(pred_np[i]), jnp.array(u_test_np[i])).item()
    print(f'Mean Testing Error: {100*sample_errors.mean():.4f}%')
    print(f'Mean Absolute Error (MAE): {tmae.item():.4f}')
    print(f'Max Absolute Error:        {tmax.item():.4f}')
    print(f'Total Training Time: {total_time:.2f} seconds')
    print(f'Model Parameters: {n_params:,}')

    
    eqx.tree_serialise_leaves(os.path.join(save_path, 'dwno_2d.eqx'), model)

    
    np.save(os.path.join(save_path, 'predictions_test.npy'), pred_np, allow_pickle=False)

    np.savez(os.path.join(save_path, 'train_metrics.npz'),
             train_loss=train_loss_log,
             test_rel_l2=test_rel_l2_log,
             test_mae=test_mae_log,
             test_max=test_max_log)

    
    sys.stdout = tee.stdout
    tee.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_param_levels', type=int, default=1)
    parser.add_argument('--num_wavelet_layers', type=int, default=6)
    parser.add_argument('--suffix', type=str, default='baseline')
    parser.add_argument("--data_subdir", type=str, default="")
    parser.add_argument('--P', type=int, default=25)
    args = parser.parse_args()
    train(args.num_param_levels, args.num_wavelet_layers, args.P, args.suffix, args.data_subdir)
