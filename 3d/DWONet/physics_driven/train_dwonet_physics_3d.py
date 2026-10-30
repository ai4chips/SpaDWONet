












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
from model_dwonet_physics_3d import (
    DWNOSeparableDeepONet3dPhysics,
    compute_branch, forward_trunk, count_params,
)
from helpers_physics_3d import hvp_fwdfwd, mse_single



ntrain = 10000
ntest = 100
batch_size = 25

seed = 42

learning_rate = 0.001
epochs = 2500
weight_decay = 1e-4

bc_weight = 1.0
pde_weight = 1.0


width = 64
level = 4
P = 25


trunk_layers = [50, 50, 50, 50, 50, 50, 50]
p = trunk_layers[-1]
R = 50
nx, ny, nz = 64, 64, 48


T_amb = 298.15
delta_T = 100.0
f_scale_branch = 0.03
f_scale_pde = 0.0125
L = 0.8


data_dir = os.path.join(_this_dir, '..', '..', '..', 'data', '3d')
base_save_path = os.path.join(_this_dir, 'results')



def rel_l2_norm(pred, true):
    num = jnp.linalg.norm((pred - true).ravel(), ord=2)
    den = jnp.linalg.norm(true.ravel(), ord=2)
    return num / den





def total_loss(model, src_branch, src_phys,
               coords_x, coords_y, coords_z,
               bc_w, pde_w, R, p, nx, ny, nz):





    Bf = compute_branch(model, src_branch)


    tx, ty, tz = model.trunk_x, model.trunk_y, model.trunk_z
    bias = model.output_bias


    u_pred = forward_trunk(Bf, tx, ty, tz, bias,
                           coords_x, coords_y, coords_z,
                           R, p, nx, ny, nz)


    v_one_x = jnp.ones_like(coords_x)
    v_one_y = jnp.ones_like(coords_y)
    v_one_z = jnp.ones_like(coords_z)

    def u_fn_x(xi):
        return forward_trunk(Bf, tx, ty, tz, bias,
                             xi, coords_y, coords_z, R, p, nx, ny, nz)

    def u_fn_y(yi):
        return forward_trunk(Bf, tx, ty, tz, bias,
                             coords_x, yi, coords_z, R, p, nx, ny, nz)

    def u_fn_z(zi):
        return forward_trunk(Bf, tx, ty, tz, bias,
                             coords_x, coords_y, zi, R, p, nx, ny, nz)

    u_x, u_xx = hvp_fwdfwd(u_fn_x, (coords_x,), (v_one_x,), return_primals=True)
    u_y, u_yy = hvp_fwdfwd(u_fn_y, (coords_y,), (v_one_y,), return_primals=True)
    u_z, u_zz = hvp_fwdfwd(u_fn_z, (coords_z,), (v_one_z,), return_primals=True)


    f_bar = src_phys / f_scale_pde

    loss_bc_top = mse_single(u_z[:, :, :, -1] - f_bar)
    loss_bc_bot = mse_single(u_z[:, :, :, 0] - 4.0 * u_pred[:, :, :, 0])
    loss_bc_left = mse_single(u_x[:, 0, :, :])
    loss_bc_right = mse_single(u_x[:, -1, :, :])
    loss_bc_front = mse_single(u_y[:, :, 0, :])
    loss_bc_back = mse_single(u_y[:, :, -1, :])
    loss_bc = (loss_bc_top + loss_bc_bot + loss_bc_left
               + loss_bc_right + loss_bc_front + loss_bc_back)


    pde_res = (u_xx[:, 1:-1, 1:-1, 1:-1]
               + u_yy[:, 1:-1, 1:-1, 1:-1]
               + u_zz[:, 1:-1, 1:-1, 1:-1])
    loss_pde = jnp.mean(pde_res ** 2)

    loss = bc_w * loss_bc + pde_w * loss_pde
    return loss, (loss_bc, loss_pde, loss_bc_top, loss_bc_bot)





def train(num_param_levels=1, num_wavelet_layers=5, suffix='baseline', data_subdir=""):
    np.random.seed(seed)
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    print(f'JAX backend: {device}')

    save_path = os.path.join(base_save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    _data_dir = os.path.join(data_dir, data_subdir) if data_subdir else data_dir


    print('Loading data...')
    source_train_all = jnp.array(
        np.load(os.path.join(_data_dir, 'f_train_physics.npy')),
        dtype=jnp.float32,
    )


    source_test = jnp.array(
        np.load(os.path.join(_data_dir, 'f_test.npy')),
        dtype=jnp.float32,
    )
    u_test = jnp.array(
        np.load(os.path.join(_data_dir, 'u_test.npy')),
        dtype=jnp.float32,
    )


    u_test_norm = (u_test - T_amb) / delta_T

    print(f'Train sources: {source_train_all.shape[0]}, '
          f'Test: {ntest}, Grid: {nx}x{ny}x{nz}, Batch: {batch_size}')
    print(f'Epochs: {epochs}  (each = 1 random batch → 1 gradient step)')
    print(f'Normalization: ΔT={delta_T}K, T_amb={T_amb}K, '
          f'f_scale_branch={f_scale_branch}, f_scale_pde={f_scale_pde}')


    coords_x = jnp.linspace(0, 1, nx, dtype=jnp.float32)
    coords_y = jnp.linspace(0, 1, ny, dtype=jnp.float32)
    coords_z = jnp.linspace(0, 0.75, nz, dtype=jnp.float32)


    print('Creating model...')
    key = jax.random.PRNGKey(0)
    model = DWNOSeparableDeepONet3dPhysics(
        trunk_layers=trunk_layers,
        R=R, nx=nx, ny=ny, nz=nz,
        width=width, level=level, P=P,
        num_param_levels=num_param_levels,
        num_wavelet_layers=num_wavelet_layers,
        key=key,
    )
    print(f'Model parameters: {count_params(model):,}')


    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=epochs,
        alpha=0.0,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))


    def loss_fn(m, src_branch, src_phys):
        return total_loss(m, src_branch, src_phys,
                          coords_x, coords_y, coords_z,
                          bc_weight, pde_weight, R, p, nx, ny, nz)

    @eqx.filter_jit
    def train_step(model, opt_state, src_branch, src_phys):
        (loss_val, aux), grads = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                model, src_branch, src_phys
            )
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, *aux

    @eqx.filter_jit
    def eval_step(model, src_branch, tgt_raw):
        pred = model(src_branch, coords_x, coords_y, coords_z)
        pred_raw = pred * delta_T + T_amb
        mse_val = jnp.mean((pred_raw - tgt_raw) ** 2)
        mae_val = jnp.mean(jnp.abs(pred_raw - tgt_raw))
        max_abs = jnp.max(jnp.abs(pred_raw - tgt_raw))
        rl2 = rel_l2_norm(pred_raw, tgt_raw)
        return mse_val, rl2, mae_val, max_abs, pred_raw


    loss_log = np.zeros(epochs, dtype=np.float32)
    bc_log = np.zeros(epochs, dtype=np.float32)
    pde_log = np.zeros(epochs, dtype=np.float32)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float32)
    test_mae_log = np.zeros(epochs, dtype=np.float32)
    test_max_log = np.zeros(epochs, dtype=np.float32)
    total_train_time = 0.0


    print(f'\n{"="*60}')
    print(f'Starting physics-driven 3D training ({epochs} iterations)')
    print(f'BC weight: {bc_weight}, PDE weight: {pde_weight}')
    print(f'{"="*60}\n')

    log_every = max(1, epochs // 100)

    for ep in range(epochs):
        t1 = time.perf_counter()

        idx = np.random.choice(ntrain, batch_size, replace=False)
        src_phys = source_train_all[idx]
        src_branch = src_phys / f_scale_branch

        model, opt_state, loss_val, loss_bc_val, loss_pde_val, \
            bc_top_val, bc_bot_val = train_step(
                model, opt_state, src_branch, src_phys
            )


        if ep % log_every == 0 or ep == epochs - 1:
            src_test_branch = source_test / f_scale_branch
            test_mse, test_rel, test_mae, test_max_abs, _ = eval_step(
                model, src_test_branch, u_test
            )
            test_rel_val = test_rel.item()
            test_mae_val = test_mae.item()
            test_max_val = test_max_abs.item()
        else:
            test_rel_val = test_rel_l2_log[max(0, ep - 1)] if ep > 0 else 1.0
            test_mae_val = test_mae_log[max(0, ep - 1)] if ep > 0 else 0.0
            test_max_val = test_max_log[max(0, ep - 1)] if ep > 0 else 0.0

        t2 = time.perf_counter()
        epoch_time = t2 - t1
        total_train_time += epoch_time

        loss_log[ep] = float(loss_val)
        bc_log[ep] = float(loss_bc_val)
        pde_log[ep] = float(loss_pde_val)
        test_rel_l2_log[ep] = test_rel_val
        test_mae_log[ep] = test_mae_val
        test_max_log[ep] = test_max_val

        if ep % log_every == 0:
            print(f'Iter {ep:5d} | Time {epoch_time:.2f}s | '
                  f'Total {loss_val.item():.6f} BC {loss_bc_val.item():.6f} '
                  f'PDE {loss_pde_val.item():.6f} | '
                  f'BC_top {bc_top_val.item():.6f} BC_bot {bc_bot_val.item():.6f} | '
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
    src_test_branch = source_test / f_scale_branch
    _, _, test_mae, test_max_abs, pred_all = eval_step(
        model, src_test_branch, u_test)
    pred_all_np = np.array(pred_all)
    u_test_np = np.array(u_test)

    sample_errors = np.zeros(ntest, dtype=np.float32)
    for i in range(ntest):
        sample_errors[i] = rel_l2_norm(
            jnp.array(pred_all_np[i]), jnp.array(u_test_np[i])
        ).item()

    mean_error = 100 * sample_errors.mean()
    print(f'Mean Testing Error: {mean_error:.4f}%')
    print(f'Mean Absolute Error (MAE): {test_mae.item():.4f} K')
    print(f'Max Absolute Error:        {test_max_abs.item():.4f} K')
    print(f'Total Training Time: {total_train_time:.2f} seconds')
    print(f'Model Parameters: {count_params(model)}')
    try:
        for d in jax.devices():
            if d.platform == 'gpu':
                mem = d.memory_stats()
                if mem:
                    peak = mem.get('peak_bytes_in_use', 0) / 1e9
                    print(f'GPU Memory Used (peak): {peak:.2f} GB')
    except Exception:
        pass


    eqx.tree_serialise_leaves(os.path.join(save_path, 'dwno_deeponet_physics_3d_bp.eqx'), model)

    np.savez(os.path.join(save_path, 'train_metrics.npz'),
             train_loss=loss_log,
             bc_loss=bc_log,
             pde_loss=pde_log,
             test_rel_l2=test_rel_l2_log,
             test_mae=test_mae_log,
             test_max=test_max_log)

    sys.stdout = tee.stdout
    tee.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_param_levels', type=int, default=1)
    parser.add_argument('--num_wavelet_layers', type=int, default=5)
    parser.add_argument('--suffix', type=str, default='baseline')
    parser.add_argument('--data_subdir', type=str, default='')
    args = parser.parse_args()
    train(args.num_param_levels, args.num_wavelet_layers, args.suffix, args.data_subdir)
