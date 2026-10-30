
import sys, os, time, argparse
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
from model_dwno_deeponet_2d import DWNOSeparableDeepONet2d, count_params


ntest = 100
ntrain = 1000
batch_size = 25     
learning_rate = 0.001
epochs = 1000
weight_decay = 1e-4

width, level, P = 64, 4, 25
trunk_layers = [50, 50, 50, 50, 50, 50, 50]
nx, ny = 64, 64

data_dir = os.path.join(_this_dir, '..', '..', 'data', '2d')
base_save_path = os.path.join(_this_dir, 'results')


def rel_l2_norm(pred, true):
    return jnp.linalg.norm((pred - true).ravel()) / jnp.linalg.norm(true.ravel())


def train(num_param_levels=1, num_wavelet_layers=6, suffix='baseline', data_subdir=""):
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    print(f'JAX backend: {device}')
    save_path = os.path.join(base_save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    _data_dir = os.path.join(data_dir, data_subdir) if data_subdir else data_dir

    print(f'DWNO-DeepONet 2D Data-Driven, k={num_param_levels}, L={num_wavelet_layers}')
    print('Loading data...')
    f_all = jnp.array(np.load(os.path.join(_data_dir, 'f_train.npy')), dtype=jnp.float64)
    u_all = jnp.array(np.load(os.path.join(_data_dir, 'u_train.npy')), dtype=jnp.float64)
    f_test = jnp.array(np.load(os.path.join(_data_dir, 'f_test.npy')), dtype=jnp.float64)
    u_test = jnp.array(np.load(os.path.join(_data_dir, 'u_test.npy')), dtype=jnp.float64)

    f_train, u_train = f_all[:ntrain], u_all[:ntrain]

    print(f'Train: {ntrain}, Test: {ntest}, Grid: {nx}x{ny}, Batch: {batch_size}')
    print(f'Width: {width}, Level: {level}, P: {P}')

    print('Creating model...')
    key = jax.random.PRNGKey(0)
    model = DWNOSeparableDeepONet2d(trunk_layers=trunk_layers, nx=nx, ny=ny,
                                     num_param_levels=num_param_levels,
                                     width=width, level=level, P=P, key=key)
    n_params = count_params(model)
    print(f'Parameters: {n_params:,}')

    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(optax.clip_by_global_norm(1.0),
                        optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(lambda m: jnp.mean((m(src)-tgt)**2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss_val

    @eqx.filter_jit
    def eval_batch(model, src, tgt):
        return model(src)

    def eval_all(model, src_all, tgt_all, eb=50):
        preds = []
        for start in range(0, src_all.shape[0], eb):
            end = min(start+eb, src_all.shape[0])
            preds.append(np.array(eval_batch(model, src_all[start:end], tgt_all[start:end])))
        pred_all = jnp.concatenate(preds, axis=0)
        d = pred_all - tgt_all
        return jnp.mean(d**2), rel_l2_norm(pred_all, tgt_all), pred_all, jnp.mean(jnp.abs(d)), jnp.max(jnp.abs(d))

    train_loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_time = 0.0
    log_every = max(1, epochs // 100)

    print(f'\n{"="*60}')
    print(f'Training ({epochs} epochs)...')
    print(f'{"="*60}\n')

    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain, batch_size, replace=False)
        model, opt_state, loss_val = train_step(model, opt_state, f_train[idx], u_train[idx])

        if ep % log_every == 0 or ep == epochs - 1:
            _, test_rel, _, test_mae, test_max_abs = eval_all(model, f_test, u_test)
            trv, tmv, txv = test_rel.item(), test_mae.item(), test_max_abs.item()
        else:
            trv = test_rel_l2_log[max(0,ep-1)] if ep>0 else 1.0
            tmv = test_mae_log[max(0,ep-1)] if ep>0 else 0.0
            txv = test_max_log[max(0,ep-1)] if ep>0 else 0.0
        t2 = time.perf_counter()
        total_time += (t2-t1)
        train_loss_log[ep] = loss_val.item()
        test_rel_l2_log[ep] = trv; test_mae_log[ep] = tmv; test_max_log[ep] = txv
        if ep % log_every == 0:
            print(f'Ep {ep:5d} | T {t2-t1:.1f}s | MSE {loss_val.item():.6f} | '
                  f'Rel.L2 {trv:.6f} MAE {tmv:.6f} MaxAbs {txv:.6f}')

    print(f'\n{"="*60}')
    print(f'Final Rel.L2: {test_rel_l2_log[-1]:.6f} ({100*test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.6f}  MaxAbs: {test_max_log[-1]:.6f}')
    print(f'Params: {n_params:,}  Time: {total_time:.1f}s')
    print(f'{"="*60}\n')

    print('Per-sample eval...')
    _, _, pred_all, test_mae, test_max_abs = eval_all(model, f_test, u_test)
    pred_np = np.array(pred_all)
    u_test_np = np.array(np.load(os.path.join(_data_dir, 'u_test.npy')))

    sample_errors = np.zeros(ntest, dtype=np.float64)
    for i in range(ntest):
        sample_errors[i] = rel_l2_norm(jnp.array(pred_np[i]), jnp.array(u_test_np[i])).item()
    print(f'Mean Error: {100*sample_errors.mean():.4f}%  MAE: {test_mae.item():.6f}  MaxAbs: {test_max_abs.item():.6f}')

    for d in jax.devices():
        if d.platform == 'gpu':
            mem = d.memory_stats()
            if mem:
                peak = mem.get('peak_bytes_in_use', 0) / 1e9
                print(f'GPU Memory Used (peak): {peak:.2f} GB')

    
    eqx.tree_serialise_leaves(os.path.join(save_path, 'dwno_deeponet_2d.eqx'), model)

    
    np.save(os.path.join(save_path, 'predictions_test.npy'), pred_np, allow_pickle=False)

    np.savez(os.path.join(save_path, 'train_metrics.npz'),
             train_loss=train_loss_log,
             test_rel_l2=test_rel_l2_log,
             test_mae=test_mae_log,
             test_max=test_max_log)

    
    sys.stdout = tee.stdout
    tee.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_param_levels', type=int, default=1)
    parser.add_argument('--num_wavelet_layers', type=int, default=6)
    parser.add_argument('--suffix', type=str, default='baseline')
    parser.add_argument("--data_subdir", type=str, default="")
    args = parser.parse_args()
    train(args.num_param_levels, args.num_wavelet_layers, args.suffix, args.data_subdir)
