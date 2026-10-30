








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
from model_spa_deeponet_3d import MLPDeepONet3dBPSS, count_params



ntrain = 500
ntest = 100
batch_size = 25
learning_rate = 0.001
epochs = 2500
seed = 42
weight_decay = 1e-4


encoder_layers = [190, 190, 190, 95]


head_layers = [95 + 2, 32, 32, 50]


trunk_layers = [50, 50, 50, 50, 50, 50, 50]
r = 30
nx, ny, nz = 64, 64, 48


T_amb = 298.15
delta_T = 100.0
f_scale = 0.03

data_dir = os.path.join(_this_dir, '..', '..', 'data', '3d')
base_save_path = os.path.join(_this_dir, 'results')



def rel_l2_norm(pred, true):
    num = jnp.linalg.norm((pred - true).ravel(), ord=2)
    den = jnp.linalg.norm(true.ravel(), ord=2)
    return num / den


def train(num_param_levels=1, num_wavelet_layers=10, suffix='baseline', data_subdir=""):
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    print(f'JAX backend: {device}')

    save_path = os.path.join(base_save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    np.random.seed(seed)
    _data_dir = os.path.join(data_dir, data_subdir) if data_subdir else data_dir


    print('Loading data...')
    f_train_all = jnp.array(
        np.load(os.path.join(_data_dir, 'f_train.npy')), dtype=jnp.float64)
    u_train_all = jnp.array(
        np.load(os.path.join(_data_dir, 'u_train.npy')), dtype=jnp.float64)
    f_test = jnp.array(
        np.load(os.path.join(_data_dir, 'f_test.npy')), dtype=jnp.float64)
    u_test = jnp.array(
        np.load(os.path.join(_data_dir, 'u_test.npy')), dtype=jnp.float64)

    f_train = f_train_all[:ntrain]
    u_train = u_train_all[:ntrain]

    f_train = f_train / f_scale
    f_test = f_test / f_scale
    u_train = (u_train - T_amb) / delta_T
    u_test = (u_test - T_amb) / delta_T

    print(f'Train: {ntrain}, Test: {ntest}, '
          f'Grid: {nx}x{ny}x{nz}, Batch: {batch_size}')
    print(f'Encoder: {[nx*ny] + encoder_layers}, head: {head_layers}, p={trunk_layers[-1]}')


    print('Creating model...')
    key = jax.random.PRNGKey(0)
    model = MLPDeepONet3dBPSS(
        encoder_layers=encoder_layers,
        head_layers=head_layers,
        trunk_layers=trunk_layers, r=r,
        nx=nx, ny=ny, nz=nz, key=key,
    )
    n_params = count_params(model)
    print(f'Parameters: {n_params:,}')


    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=epochs, alpha=0.0)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(src) - tgt) ** 2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def eval_batch(model, src, tgt):
        return model(src)

    def eval_all(model, src_all, tgt_all, eval_batch_size=25,
                 T_amb=0.0, delta_T=1.0):
        n_total = src_all.shape[0]
        preds = []
        for start in range(0, n_total, eval_batch_size):
            end = min(start + eval_batch_size, n_total)
            pred = eval_batch(model, src_all[start:end], tgt_all[start:end])
            preds.append(np.array(pred))
        pred_all = jnp.concatenate(preds, axis=0)
        pred_raw = pred_all * delta_T + T_amb
        tgt_raw = tgt_all * delta_T + T_amb
        diff = pred_raw - tgt_raw
        mse_val = jnp.mean(diff ** 2)
        mae_val = jnp.mean(jnp.abs(diff))
        max_abs = jnp.max(jnp.abs(diff))
        rl2 = rel_l2_norm(pred_raw, tgt_raw)
        return mse_val, rl2, pred_raw, mae_val, max_abs


    train_loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_train_time = 0.0

    print(f'\n{"="*60}')
    print(f'Training ({epochs} epochs, 1 random batch/epoch)')
    print(f'{"="*60}\n')

    log_every = max(1, epochs // 100)

    for ep in range(epochs):
        t1 = time.perf_counter()

        idx = np.random.choice(ntrain, batch_size, replace=False)
        src_batch = f_train[idx]
        tgt_batch = u_train[idx]

        model, opt_state, loss_val = train_step(
            model, opt_state, src_batch, tgt_batch)

        if ep % log_every == 0 or ep == epochs - 1:
            test_mse, test_rel, _, test_mae, test_max_abs = eval_all(
                model, f_test, u_test, T_amb=T_amb, delta_T=delta_T)
            test_rel_val = test_rel.item()
            test_mae_val = test_mae.item()
            test_max_val = test_max_abs.item()
        else:
            test_rel_val = test_rel_l2_log[max(0, ep - 1)] if ep > 0 else 1.0
            test_mae_val = test_mae_log[max(0, ep - 1)] if ep > 0 else 0.0
            test_max_val = test_max_log[max(0, ep - 1)] if ep > 0 else 0.0

        t2 = time.perf_counter()
        total_train_time += (t2 - t1)

        train_loss_log[ep] = loss_val.item()
        test_rel_l2_log[ep] = test_rel_val
        test_mae_log[ep] = test_mae_val
        test_max_log[ep] = test_max_val

        if ep % log_every == 0:
            print(f'Iter {ep:5d} | Time {t2-t1:.2f}s | '
                  f'Train MSE {loss_val.item():.6f} | '
                  f'Test Rel.L2 {test_rel_val:.6f} '
                  f'MAE {test_mae_val:.4f}K MaxAbs {test_max_val:.4f}K')

    print(f'\n{"="*60}')
    print(f'Training complete. Total time: {total_train_time:.2f}s')
    print(f'Final Test Rel.L2: {test_rel_l2_log[-1]:.6f} '
          f'({100 * test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.4f}K  '
          f'MaxAbs: {test_max_log[-1]:.4f}K')
    print(f'{"="*60}\n')


    print('Computing per-sample test error...')
    _, _, pred_all, test_mae, test_max_abs = eval_all(
        model, f_test, u_test, T_amb=T_amb, delta_T=delta_T)
    pred_all_np = np.array(pred_all)
    prediction_path = os.path.join(_this_dir, '..', 'predictions', 'spa_deeponet_predictions.npy')
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    np.save(prediction_path, pred_all_np, allow_pickle=False)
    print(f'Benchmark predictions saved to: {prediction_path}')
    u_test_np = np.array(np.load(os.path.join(_data_dir, 'u_test.npy')))

    sample_errors = np.zeros(ntest, dtype=np.float64)
    for i in range(ntest):
        sample_errors[i] = rel_l2_norm(
            jnp.array(pred_all_np[i]), jnp.array(u_test_np[i])).item()

    mean_error = 100 * sample_errors.mean()
    print(f'Mean Testing Error: {mean_error:.4f}%')
    print(f'Mean Absolute Error (MAE): {test_mae.item():.4f} K')
    print(f'Max Absolute Error:        {test_max_abs.item():.4f} K')
    print(f'Total Training Time: {total_train_time:.2f} seconds')
    print(f'Model Parameters: {n_params:,}')


    eqx.tree_serialise_leaves(os.path.join(save_path, 'mlp_bpss_3d.eqx'), model)

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
    parser.add_argument('--num_wavelet_layers', type=int, default=10)
    parser.add_argument('--suffix', type=str, default='mlp_bpss')
    parser.add_argument('--data_subdir', type=str, default='')
    args = parser.parse_args()
    train(args.num_param_levels, args.num_wavelet_layers, args.suffix, args.data_subdir)
