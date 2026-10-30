










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


_dd_dir = os.path.join(_this_dir, '..', 'data_driven')
sys.path.insert(0, _dd_dir)
from model_spadwonet_3d import DWNOSeparableDeepONet3d, count_params


_pd_dir = os.path.join(_this_dir, '..', 'physics_driven')
sys.path.insert(0, _pd_dir)
from model_spadwonet_physics_3d import DWNOSeparableDeepONet3dPhysics



ntrain = 500
ntest = 100
batch_size = 25

seed = 42
learning_rate = 0.001
epochs = 2500
weight_decay = 1e-4


width, level, P = 64, 4, 25


trunk_layers = [50, 50, 50, 50, 50, 50, 50]
R = 50
nx, ny, nz = 64, 64, 48

T_amb, delta_T = 298.15, 100.0
f_scale_branch = 0.03

data_dir = os.path.join(_this_dir, '..', '..', '..', 'data', '3d')
phys_model_path = os.path.join(
    _this_dir, '..', 'physics_driven', 'results', 'baseline',
    'dwno_deeponet_physics_3d_bpss.eqx',
)
base_save_path = os.path.join(_this_dir, 'results')



def rel_l2_norm(pred, true):
    return jnp.linalg.norm((pred - true).ravel()) / jnp.linalg.norm(true.ravel())


def train(num_param_levels=1, num_wavelet_layers=10, ntrain_val=None, suffix='baseline', data_subdir=""):
    if ntrain_val is None:
        ntrain_val = ntrain
    np.random.seed(seed)
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    print(f'JAX backend: {device}')

    save_path = os.path.join(base_save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    _data_dir = os.path.join(data_dir, data_subdir) if data_subdir else data_dir

    k = num_param_levels
    L = num_wavelet_layers




    print('Loading physics model...')
    phys_key = jax.random.PRNGKey(0)
    phys_model = DWNOSeparableDeepONet3dPhysics(
        trunk_layers=trunk_layers, R=R, nx=nx, ny=ny, nz=nz,
        width=width, level=level, P=P,
        num_param_levels=1, num_wavelet_layers=5, key=phys_key,
    )
    if os.path.exists(phys_model_path):
        phys_model = eqx.tree_deserialise_leaves(phys_model_path, phys_model)
        print('  Loaded pre-trained weights')
    else:
        print(f'  WARNING: no saved weights at {phys_model_path}, using random init')
    print(f'  Physics model parameters: {count_params(phys_model):,}')

    @eqx.filter_jit
    def phys_forward(model, src):
        return model(src)




    print('Loading data...')
    s_all = jnp.array(np.load(os.path.join(_data_dir, 'f_train.npy')), dtype=jnp.float64)
    u_all = jnp.array(np.load(os.path.join(_data_dir, 'u_train.npy')), dtype=jnp.float64)
    s_test = jnp.array(np.load(os.path.join(_data_dir, 'f_test.npy')), dtype=jnp.float64)
    u_test = jnp.array(np.load(os.path.join(_data_dir, 'u_test.npy')), dtype=jnp.float64)

    s_train, u_train = s_all[:ntrain_val], u_all[:ntrain_val]
    s_train_norm = s_train / f_scale_branch
    s_test_norm = s_test / f_scale_branch
    u_train_norm = (u_train - T_amb) / delta_T
    u_test_norm = (u_test - T_amb) / delta_T




    print('Computing physics predictions and residuals...')
    eb = 25
    phys_train = []
    for start in range(0, ntrain_val, eb):
        end = min(start + eb, ntrain_val)
        phys_train.append(np.array(phys_forward(phys_model, s_train_norm[start:end])))
    phys_train = jnp.concatenate(phys_train, axis=0)
    r_train = u_train_norm - phys_train

    phys_test = []
    for start in range(0, ntest, eb):
        end = min(start + eb, ntest)
        phys_test.append(np.array(phys_forward(phys_model, s_test_norm[start:end])))
    phys_test = jnp.concatenate(phys_test, axis=0)

    print(f'  Residual range: [{float(r_train.min()):.4f}, {float(r_train.max()):.4f}] (norm)')
    print(f'  Residual std:   {float(r_train.std()):.4f}')


    phys_test_raw = phys_test * delta_T + T_amb
    u_test_raw = u_test
    phys_rl2 = rel_l2_norm(phys_test_raw, u_test_raw)
    phys_mae = jnp.mean(jnp.abs(phys_test_raw - u_test_raw))
    phys_maxabs = jnp.max(jnp.abs(phys_test_raw - u_test_raw))
    print(f'\nPhysics-only baseline:')
    print(f'  Rel.L2: {phys_rl2.item():.6f} ({100*phys_rl2.item():.4f}%)')
    print(f'  MAE:    {phys_mae.item():.4f} K')
    print(f'  MaxAbs: {phys_maxabs.item():.4f} K')




    print(f'\n{"="*60}')
    print(f'Training residual model (k={k}, L={L})')
    print(f'Train: {ntrain_val}, Test: {ntest}, Batch: {batch_size}')
    print(f'{"="*60}\n')

    key = jax.random.PRNGKey(42)
    res_model = DWNOSeparableDeepONet3d(
        trunk_layers=trunk_layers, R=R, nx=nx, ny=ny, nz=nz,
        width=width, level=level, P=P,
        num_param_levels=k, num_wavelet_layers=L, key=key,
    )
    print(f'Residual model parameters: {count_params(res_model):,}')

    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(res_model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(src) - tgt) ** 2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def eval_batch(model, src):
        return model(src)

    def eval_combined(r_model, src_all, phys_all, tgt_raw_all, eb_ev=25):
        preds = []
        for start in range(0, src_all.shape[0], eb_ev):
            end = min(start + eb_ev, src_all.shape[0])
            r_pred = eval_batch(r_model, src_all[start:end])
            u_raw = (phys_all[start:end] + r_pred) * delta_T + T_amb
            preds.append(np.array(u_raw))
        pred_all = jnp.concatenate(preds, axis=0)
        d = pred_all - tgt_raw_all
        return jnp.mean(d**2), rel_l2_norm(pred_all, tgt_raw_all), jnp.mean(jnp.abs(d)), jnp.max(jnp.abs(d)), pred_all

    train_loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_time = 0.0
    log_every = max(1, epochs // 100)

    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain_val, batch_size, replace=False)
        res_model, opt_state, loss_val = train_step(
            res_model, opt_state, s_train_norm[idx], r_train[idx])

        if ep % log_every == 0 or ep == epochs - 1:
            _, test_rel, test_mae, test_max_abs, _ = eval_combined(
                res_model, s_test_norm, phys_test, u_test_raw)
            trv, tmv, txv = test_rel.item(), test_mae.item(), test_max_abs.item()
        else:
            trv = test_rel_l2_log[max(0, ep - 1)] if ep > 0 else 1.0
            tmv = test_mae_log[max(0, ep - 1)] if ep > 0 else 0.0
            txv = test_max_log[max(0, ep - 1)] if ep > 0 else 0.0

        t2 = time.perf_counter()
        total_time += (t2 - t1)
        train_loss_log[ep] = float(loss_val)
        test_rel_l2_log[ep] = trv; test_mae_log[ep] = tmv; test_max_log[ep] = txv

        if ep % log_every == 0:
            print(f'Iter {ep:5d} | Time {t2-t1:.2f}s | '
                  f'Train MSE {loss_val.item():.6f} | '
                  f'Test Rel.L2 {trv:.6f} MAE {tmv:.4f}K MaxAbs {txv:.4f}K')

    print(f'\n{"="*60}')
    print(f'Training complete. Total time: {total_time:.1f}s')
    print(f'Final Test Rel.L2: {test_rel_l2_log[-1]:.6f} ({100*test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.4f}K  MaxAbs: {test_max_log[-1]:.4f}K')
    print(f'{"="*60}\n')




    print('Computing per-sample test errors...')
    _, _, _, _, pred_combined = eval_combined(res_model, s_test_norm, phys_test, u_test_raw)
    pred_np = np.array(pred_combined)
    phys_np = np.array(phys_test_raw)
    u_test_np = np.array(u_test_raw)

    combined_errors = np.zeros(ntest, dtype=np.float64)
    phys_errors = np.zeros(ntest, dtype=np.float64)
    for i in range(ntest):
        combined_errors[i] = rel_l2_norm(jnp.array(pred_np[i]), jnp.array(u_test_np[i])).item()
        phys_errors[i] = rel_l2_norm(jnp.array(phys_np[i]), jnp.array(u_test_np[i])).item()

    print(f'\nPhysics-only:       Rel.L2 = {100*phys_errors.mean():.4f}%')
    print(f'Residual combined:  Rel.L2 = {100*combined_errors.mean():.4f}%')
    print(f'Improvement:        {100*(phys_errors.mean()-combined_errors.mean()):.4f}%')


    eqx.tree_serialise_leaves(os.path.join(save_path, 'residual_3d.eqx'), res_model)

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
    parser.add_argument('--suffix', type=str, default='baseline')
    parser.add_argument('--data_subdir', type=str, default='')
    parser.add_argument('--ntrain', type=int, default=500)
    args = parser.parse_args()
    train(args.num_param_levels, args.num_wavelet_layers, args.ntrain, args.suffix, args.data_subdir)
