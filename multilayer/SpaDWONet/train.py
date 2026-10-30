


import sys, os, time, argparse
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp, numpy as np, optax, equinox as eqx

class Tee:
    def __init__(self, fname, mode='w'):
        self.file = open(fname, mode); self.stdout = sys.stdout
    def write(self, data): self.file.write(data); self.stdout.write(data)
    def flush(self): self.file.flush(); self.stdout.flush()
    def close(self): self.file.close()

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = _this_dir
sys.path.insert(0, _src_dir)
from model import DWNOSeparableDeepONet3d, count_params

ntrain, ntest = 500, 100; batch_size = 25; learning_rate = 0.001; epochs = 2500
weight_decay, seed = 1e-4, 42
width, level, P = 64, 4, 25; trunk_layers = [50]*7; R, nx, ny, nz = 50, 64, 64, 48
T_amb, delta_T, f_scale = 298.15, 100.0, 0.03
data_dir = os.path.join(_this_dir, '..', '..', 'data', 'multilayer')
base_save_path = os.path.join(_this_dir, 'results')

def rel_l2_norm(pred, true):
    return jnp.linalg.norm((pred-true).ravel()) / jnp.linalg.norm(true.ravel())

def train(z_layer):
    suffix = f'z{z_layer}'
    save_path = os.path.join(base_save_path, suffix); os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w'); sys.stdout = tee
    np.random.seed(seed)
    print(f'DWNO-BPSS z={z_layer}')
    f_train = jnp.array(np.load(os.path.join(data_dir, 'f_train.npy')), dtype=jnp.float32)[:ntrain] / f_scale
    u_train = jnp.array(np.load(os.path.join(data_dir, f'u_train_z{z_layer}.npy')), dtype=jnp.float32)[:ntrain]
    u_train = (u_train - T_amb) / delta_T
    f_test = jnp.array(np.load(os.path.join(data_dir, 'f_test.npy')), dtype=jnp.float32)[:ntest] / f_scale
    u_test = jnp.array(np.load(os.path.join(data_dir, f'u_test_z{z_layer}.npy')), dtype=jnp.float32)[:ntest]
    u_test = (u_test - T_amb) / delta_T
    print(f'Train: {ntrain}, Test: {ntest}, Grid: {nx}x{ny}x{nz}, Batch: {batch_size}')
    key = jax.random.PRNGKey(0)
    model = DWNOSeparableDeepONet3d(trunk_layers=trunk_layers, R=R, nx=nx, ny=ny, nz=nz, width=width, level=level, P=P, num_param_levels=1, num_wavelet_layers=10, key=key)
    print(f'Parameters: {count_params(model):,}')
    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(m, os_, s, t):
        loss, grads = eqx.filter_value_and_grad(lambda x: jnp.mean((x(s)-t)**2))(m)
        u, os_ = optim.update(grads, os_, m); m = eqx.apply_updates(m, u); return m, os_, loss

    @eqx.filter_jit
    def eval_batch(m, s): return m(s)

    def eval_all(m, sa, ta, eb=25):
        preds = [np.array(eval_batch(m, sa[s:s+eb])) for s in range(0, len(sa), eb)]
        pa = jnp.concatenate(preds, 0) * delta_T + T_amb
        ta = ta * delta_T + T_amb; d = pa - ta
        return jnp.mean(d**2), rel_l2_norm(pa,ta), jnp.mean(jnp.abs(d)), jnp.max(jnp.abs(d)), pa

    loss_log = np.zeros(epochs); rl2_log = np.zeros(epochs); mae_log = np.zeros(epochs); mx_log = np.zeros(epochs)
    tt = 0.0; le = max(1, epochs//100)
    print(f'\n{"="*60}\nTraining ({epochs} epochs)\n{"="*60}\n')
    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain, batch_size, replace=False)
        model, opt_state, loss_val = train_step(model, opt_state, f_train[idx], u_train[idx])
        if ep % le == 0 or ep == epochs-1:
            _, rl2, mae, mx, _ = eval_all(model, f_test, u_test); tr, tm, tx = rl2.item(), mae.item(), mx.item()
        else: tr = rl2_log[max(0,ep-1)] if ep>0 else 1.0; tm = mae_log[max(0,ep-1)] if ep>0 else 0.0; tx = mx_log[max(0,ep-1)] if ep>0 else 0.0
        t2 = time.perf_counter(); tt += t2-t1
        loss_log[ep]=float(loss_val); rl2_log[ep]=tr; mae_log[ep]=tm; mx_log[ep]=tx
        if ep % le == 0: print(f'Iter {ep:5d} | Time {t2-t1:.2f}s | Train MSE {loss_val.item():.6f} | Test Rel.L2 {tr:.6f} MAE {tm:.4f}K MaxAbs {tx:.4f}K')
    print(f'\n{"="*60}\nTraining complete. Total: {tt:.1f}s\nFinal Rel.L2: {rl2_log[-1]:.6f} ({100*rl2_log[-1]:.4f}%)\nFinal MAE: {mae_log[-1]:.4f}K  MaxAbs: {mx_log[-1]:.4f}K\n{"="*60}\n')
    _, _, mae, mx, pa = eval_all(model, f_test, u_test)
    u_true = np.load(os.path.join(data_dir, f'u_test_z{z_layer}.npy'))[:ntest]
    errs = np.array([rel_l2_norm(jnp.array(pa[i]), jnp.array(u_true[i])).item() for i in range(ntest)])
    print(f'Mean Testing Error: {100*errs.mean():.4f}%\nMAE: {mae.item():.4f}K  MaxAbs: {mx.item():.4f}K\nTotal Time: {tt:.1f}s  Params: {count_params(model):,}')
    eqx.tree_serialise_leaves(os.path.join(save_path, f'dwno_bpss_z{z_layer}.eqqx'), model)
    np.savez(os.path.join(save_path, 'train_metrics.npz'), train_loss=loss_log, test_rel_l2=rl2_log, test_mae=mae_log, test_max=mx_log)
    sys.stdout = tee.stdout; tee.close()
    print(f'Saved to {save_path}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--z', type=int, required=True, choices=[1,4,8,12,16]); args = parser.parse_args(); train(args.z)
