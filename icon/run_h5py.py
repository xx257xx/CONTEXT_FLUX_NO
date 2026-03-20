# run_paper_cubic.py
# ------------------------------------------------------------
# Paper-repro default runner for Cubic Conservation Law (ICON-style)
# - Uses your HDF5 dataset:
#   /home/taeyoungkim/CONTEXT_FLUX_NO/data/cubic_no_source_train_1000_100.hdf5
# - Uses your h5loader.py (you said you saved my H5 DataProvider there)
# - Sets FLAGS defaults to match the paper-like setting:
#   Nx=100 tokens, tau=0.1, t_init_window=0.4, pairs_per_operator=10000,
#   transformer dim=256, heads=8, layers=6, batch=16, total steps=1e6
# ------------------------------------------------------------

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import inspect
from pprint import pprint
from datetime import datetime
import pickle


import pytz
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")

import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import haiku as hk

from absl import app, flags, logging

import utils
from utils import load_json

import models
import plot

# IMPORTANT: use your HDF5 loader
from h5loader import DataProvider


gpus = tf.config.list_physical_devices(device_type="GPU")
print("TF GPUs:", gpus, flush=True)
print("JAX devices:", jax.devices(), flush=True)


def _make_dataprovider(cls, **kwargs):
    """
    Create DataProvider but only pass kwargs that the class actually accepts.
    This makes the runner robust if your h5loader.DataProvider signature differs a bit.
    """
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    missing = [k for k in kwargs.keys() if k not in filtered]
    if missing:
        print(f"[INFO] DataProvider ignored unsupported kwargs: {missing}", flush=True)
    return cls(**filtered)


class Runner:
    def __init__(
        self,
        seed,
        prompt_dim,
        query_dim,
        qoi_v_dim,
        hidden_dim,
        num_heads,
        num_layers,
        optimizer,
        initializer="glorot_uniform",
        devices=jax.devices(),
    ):
        self.seed = seed
        self.prompt_dim = prompt_dim
        self.query_dim = query_dim
        self.qoi_v_dim = qoi_v_dim

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.initializer = initializer

        self.devices = devices
        self.num_devices = len(devices)

        self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

        self.params, self.predict_fn, self.loss_fn = self.build_basic_fn()  # no batch
        self.opt_state = optimizer.init(self.params)
        utils.print_pytree(self.params)

        # batch (single-device) functions
        self.predict_batch_fn = jax.jit(
            jax.vmap(self.predict_fn, in_axes=[None, None, 0, 0, 0], out_axes=0)
        )
        self.loss_batch_fn = jax.jit(
            jax.vmap(self.loss_fn, in_axes=[None, None, 0, 0, 0, 0, 0], out_axes=0)
        )
        self.loss_batch_ave_fn = jax.jit(lambda *args, **kwargs: jnp.mean(self.loss_batch_fn(*args, **kwargs)))

        # multi-device (pmap)
        self.params = jax.device_put_replicated(self.params, devices)
        self.opt_state = jax.device_put_replicated(self.opt_state, devices)
        self.predict_pmap_batch_fn = jax.pmap(self.predict_batch_fn, axis_name="devices")
        self.loss_pmap_batch_fn = jax.pmap(self.loss_batch_fn, axis_name="devices")
        self.loss_pmap_batch_ave_fn = jax.pmap(self.loss_batch_ave_fn, axis_name="devices")
        self.train_iter = utils.get_train_iter_pmap(self.loss_batch_ave_fn, optimizer)

        self.train_step = 0

    def next_key(self):
        return einshape("i->ji", next(self.rng), j=self.num_devices)

    def build_basic_fn(self):
        def f(prompt, mask, query):
            net = models.SolverModel(
                q_size=self.hidden_dim,
                kv_size=self.hidden_dim,
                qoi_v_size=self.qoi_v_dim,
                QK_size=self.hidden_dim,
                V_size=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                initializer=self.initializer,
            )
            return net(prompt, mask, query)

        f = hk.transform(f)
        prompt = jnp.ones((10, self.prompt_dim))
        mask = jnp.ones((10,))
        query = jnp.ones((5, self.query_dim))
        params = f.init(next(self.rng), prompt, mask, query)

        @jax.jit
        def predict_fn(params, rng_key, prompt, mask, query):
            return f.apply(params, rng_key, prompt, mask, query)

        @jax.jit
        def loss_fn(params, rng_key, prompt, mask, query, query_mask, ground_truth):
            out = predict_fn(params, rng_key, prompt, mask, query)
            loss = jnp.mean((out - ground_truth) ** 2, where=query_mask[..., None])
            return loss

        return params, predict_fn, loss_fn

    def iter(self, prompt, mask, query, query_mask, ground_truth, use_list=False):
        self.params, self.opt_state = self.train_iter(
            self.params, self.next_key(), self.opt_state, prompt, mask, query, query_mask, ground_truth
        )
        self.train_step += 1

    def get_loss(self, prompt, mask, query, query_mask, ground_truth, use_list=False):
        losses = self.loss_pmap_batch_fn(self.params, self.next_key(), prompt, mask, query, query_mask, ground_truth)
        return losses

    def get_pred(self, prompt, mask, query, use_list=False):
        pred = self.predict_pmap_batch_fn(self.params, self.next_key(), prompt, mask, query)
        return pred


def run_train():
    stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp:", stamp, flush=True)

    train_warmup_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_warmup_percent // 100
    train_decay_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_decay_percent // 100
    print("train_warmup_steps =", train_warmup_steps, flush=True)
    print("train_decay_steps  =", train_decay_steps, flush=True)

    train_data_dirs = FLAGS.train_data_dirs
    test_data_dirs = FLAGS.train_data_dirs if FLAGS.test_data_dirs is None else FLAGS.test_data_dirs

    train_file_names = [f"{d}/{g}" for d in train_data_dirs for g in FLAGS.train_data_globs]
    test_file_names = [f"{d}/{g}" for d in test_data_dirs for g in FLAGS.test_data_globs]

    print("train_file_names:", flush=True)
    pprint(train_file_names)
    print("test_file_names:", flush=True)
    pprint(test_file_names)

    train_config = load_json(FLAGS.train_config_filename)
    test_config = train_config if FLAGS.test_config_filename is None else load_json(FLAGS.test_config_filename)

    print("train_config:", flush=True)
    pprint(train_config)
    print("test_config:", flush=True)
    pprint(test_config)

    optimizer = utils.get_scheduled_adamw(
        peak_lr=FLAGS.train_peak_lr,
        end_lr=FLAGS.train_end_lr,
        warmup_steps=train_warmup_steps,
        decay_steps=train_decay_steps,
        gnorm_clip=FLAGS.train_gnorm_clip,
        weight_decay=FLAGS.train_weight_decay,
    )

    # --- DataProviders (HDF5) ---
    train_data = _make_dataprovider(
        DataProvider,
        seed=FLAGS.seed + 1,
        demo_num=FLAGS.demo_num,
        cond_len=FLAGS.cond_len,
        qoi_len=FLAGS.qoi_len,
        batch_size=FLAGS.train_batch_size,
        shuffle_buffer_size=FLAGS.train_shuffle_buffer_size,
        file_names=train_file_names,
        k_dim=FLAGS.k_dim,
        v_dim=FLAGS.v_dim,
        config=train_config,
        select="sequential",
        k_mode=FLAGS.k_mode,
        deterministic=FLAGS.deterministic,
        # paper-repro pair sampling (if your h5loader supports these)
        tau=FLAGS.tau,
        t_init_window=FLAGS.t_init_window,
        pairs_per_operator=FLAGS.pairs_per_operator,
        direction=FLAGS.direction,
    )

    test_data = _make_dataprovider(
        DataProvider,
        seed=FLAGS.seed + 10,
        demo_num=FLAGS.demo_num,
        cond_len=FLAGS.cond_len,
        qoi_len=FLAGS.qoi_len,
        batch_size=FLAGS.train_batch_size,
        shuffle_buffer_size=FLAGS.train_shuffle_buffer_size,
        file_names=test_file_names,
        k_dim=FLAGS.k_dim,
        v_dim=FLAGS.v_dim,
        config=test_config,
        select="sequential",
        k_mode=FLAGS.k_mode,
        deterministic=FLAGS.deterministic,
        tau=FLAGS.tau,
        t_init_window=FLAGS.t_init_window,
        pairs_per_operator=FLAGS.pairs_per_operator,
        direction=FLAGS.direction,
    )

    # Example batch
    exm_equation, exm_prompt, exm_mask, exm_query, exm_query_mask, exm_ground_truth, dummy = train_data.get_next_data(
        decode_equation=True, list_size=0
    )
    train_data.pretty_print(exm_equation, exm_prompt, exm_mask, exm_query, exm_query_mask, exm_ground_truth)

    runner = Runner(
        seed=FLAGS.seed,
        prompt_dim=exm_prompt.shape[-1],
        query_dim=exm_query.shape[-1],
        qoi_v_dim=FLAGS.qoi_v_dim,
        hidden_dim=FLAGS.hidden_dim,
        num_heads=FLAGS.num_heads,
        num_layers=FLAGS.num_layers,
        optimizer=optimizer,
        initializer=FLAGS.initializer,
    )

    # TensorBoard
    if FLAGS.tfboard:
        results_dir = f"./results/{FLAGS.problem}/" + stamp
        file_writer = tf.summary.create_file_writer(results_dir)
        ckpt_dir = f"./check_points/{FLAGS.problem}/" + stamp
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_dir = f"./check_points/{FLAGS.problem}/" + stamp
    os.makedirs(ckpt_dir, exist_ok=True)

    def _unreplicate(pytree):
        # pmap replicate된 (n_devices, ...) -> 첫 번째 디바이스만 떼기
        return jax.tree_util.tree_map(lambda x: x[0], pytree)

    def save_ckpt(step: int):
        state = {
            "step": step,
            "params": jax.device_get(_unreplicate(runner.params)),
            "opt_state": jax.device_get(_unreplicate(runner.opt_state)),
            "flags": {k: v._value for k, v in FLAGS.__flags.items()},
        }
        path = os.path.join(ckpt_dir, f"ckpt_{step:08d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CKPT] saved: {path}", flush=True)


    utils.timer.tic("since last print")

    total_steps = FLAGS.epochs * FLAGS.steps_per_epoch
    for _ in range(total_steps + 1):
        # loss logging
        if runner.train_step % FLAGS.loss_freq == 0:
            utils.timer.toc("since last print")
            utils.timer.tic("since last print")

            _, prompt, mask, query, query_mask, ground_truth, dummy = train_data.get_next_data(list_size=FLAGS.list_size)
            train_loss = runner.get_loss(prompt, mask, query, query_mask, ground_truth)
            train_loss_mean = float(jnp.mean(train_loss)); train_loss_std = float(jnp.std(train_loss))

            equation, prompt_t, mask_t, query_t, query_mask_t, ground_truth_t, dummy   = test_data.get_next_data(decode_equation=True, list_size=FLAGS.list_size)
            test_loss = runner.get_loss(prompt_t, mask_t, query_t, query_mask_t, ground_truth_t)
            test_loss_mean = float(jnp.mean(test_loss)); test_loss_std = float(jnp.std(test_loss))

            print(
                f"step: {runner.train_step}, "
                f"train loss: {train_loss_mean:.6f}+-{train_loss_std:.6f}, "
                f"test loss: {test_loss_mean:.6f}+-{test_loss_std:.6f}",
                flush=True,
            )
            print("eqn[0:3]:", equation[0:3], flush=True)

            if FLAGS.tfboard:
                with file_writer.as_default():
                    tf.summary.scalar("loss/train_loss", train_loss_mean, step=runner.train_step)
                    tf.summary.scalar("loss/test_loss", test_loss_mean, step=runner.train_step)

            # checkpoint save
            if (runner.train_step % FLAGS.ckpt_freq) == 0:
                save_ckpt(runner.train_step)
                if FLAGS.tfboard:
                    file_writer.flush()

        # plot
        if FLAGS.tfboard and (runner.train_step % FLAGS.plot_freq == 0):
            equation, prompt, mask, query, query_mask, ground_truth,dummy = test_data.get_next_data(
                decode_equation=True, list_size=FLAGS.list_size
            )
            pred = runner.get_pred(prompt, mask, query)

            plot_num = FLAGS.plot_num if FLAGS.plot_num is not None else FLAGS.train_batch_size
            with file_writer.as_default():
                for fij in range(plot_num):
                    fi = fij // (FLAGS.train_batch_size // runner.num_devices)
                    fj = fij % (FLAGS.train_batch_size // runner.num_devices)
                    fig = plot.plot_all_in_one(
                        equation[fij],
                        prompt[fi, fj],
                        mask[fi, fj],
                        query[fi, fj],
                        query_mask[fi, fj],
                        ground_truth[fi, fj],
                        pred[fi, fj],
                        demo_num=FLAGS.demo_num,
                        k_dim=FLAGS.k_dim,
                        v_dim=FLAGS.v_dim,
                        k_mode=FLAGS.k_mode,
                    )
                    tf.summary.image(f"test case {fi}-{fj}", fig, step=runner.train_step)

        # training step
        _, prompt, mask, query, query_mask, ground_truth,dummy = train_data.get_next_data(list_size=FLAGS.list_size)
        runner.iter(prompt, mask, query, query_mask, ground_truth)

        # time estimate (optional)
        if runner.train_step == 100:
            utils.timer.tic("time estimate")
        if runner.train_step > 0 and (runner.train_step % FLAGS.time_freq == 0):
            ratio = runner.train_step / total_steps
            utils.timer.estimate_time("time estimate", ratio)


def main(argv):
    for key, value in FLAGS.__flags.items():
        print(value.name, ": ", value._value, flush=True)

    tf.random.set_seed(FLAGS.seed + 123456)

    if FLAGS.main == "train":
        run_train()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # ---- basics ----
    flags.DEFINE_enum("main", "train", ["test", "train"], "train or test")
    flags.DEFINE_boolean("tfboard", False, "dump into tfboard")
    flags.DEFINE_boolean("deterministic", True, "deterministic mode")
    flags.DEFINE_string("problem", "cubic_conservation_paper", "problem name for logs")
    flags.DEFINE_integer("seed", 42, "random seed")
    flags.DEFINE_integer("ckpt_freq", 1000, "checkpoint save frequency (steps)")


    # ---- dataset paths (paper repro defaults) ----
    flags.DEFINE_list(
        "train_data_dirs",
        ["/home/taeyoungkim/CONTEXT_FLUX_NO/data"],
        "directories of training data",
    )
    flags.DEFINE_list(
        "train_data_globs",
        ["cubic_no_source_train_1000_100.hdf5"],
        "filename(s) for training data",
    )
    flags.DEFINE_list("test_data_dirs", None, "directories of testing data (None -> same as train)")
    flags.DEFINE_list(
        "test_data_globs",
        ["cubic_no_source_train_1000_100.hdf5"],
        "filename(s) for testing data",
    )

    # configs (keep your existing jsons)
    flags.DEFINE_string("train_config_filename", "train_config.json", "config file for training")
    flags.DEFINE_string("test_config_filename", None, "config file for testing (None -> same as train)")

    # ---- paper pair extraction defaults ----
    flags.DEFINE_float("tau", 0.005, "time gap tau (paper: 0.1)")
    flags.DEFINE_float("t_init_window", 0.4, "starting window for pairs (paper: 0.4)")
    flags.DEFINE_integer("pairs_per_operator", 10000, "pairs per operator after downsampling (paper: 10000)")
    flags.DEFINE_enum("direction", "both", ["forward", "backward", "both"], "pair direction (paper-like: both)")

    # ---- in-context setup (paper-like) ----
    flags.DEFINE_integer("demo_num", 20, "number of demos (paper: 5)")
    flags.DEFINE_integer("cond_len", 100, "tokens per condition function (paper: Nx=100)")
    flags.DEFINE_integer("qoi_len", 100, "tokens per QoI function (paper: Nx=100)")
    flags.DEFINE_integer("k_dim", 1, "key dim (paper: x only)")
    flags.DEFINE_integer("v_dim", 1, "value dim (paper: u only)")
    flags.DEFINE_string("k_mode", "naive", "mode for keys (paper: x only)")
    flags.DEFINE_integer("query_len_max", 100, "max query length (paper: 100)")
    flags.DEFINE_integer("qoi_v_dim", 1, "output dim (paper: 1)")

    # ---- model (paper-like) ----
    flags.DEFINE_integer("hidden_dim", 256, "model dim (paper: 256)")
    flags.DEFINE_integer("num_heads", 8, "num attention heads (paper: 8)")
    flags.DEFINE_integer("num_layers", 6, "num transformer layers (paper: 6)")
    flags.DEFINE_string("initializer", "glorot_uniform", "initializer")

    # ---- training schedule (paper-like) ----
    flags.DEFINE_integer("train_batch_size", 16, "batch size (paper-like: 8 forward + 8 backward)")
    flags.DEFINE_integer("train_shuffle_buffer_size", 1000, "shuffle buffer size")
    flags.DEFINE_float("train_peak_lr", 1e-4, "peak learning rate (paper: 1e-4)")
    flags.DEFINE_float("train_end_lr", 0.0, "ending learning rate (paper: 0)")
    flags.DEFINE_integer("train_warmup_percent", 10, "warmup percent (paper: 10)")
    flags.DEFINE_integer("train_decay_percent", 100, "decay percent (paper: 100)")
    flags.DEFINE_float("train_gnorm_clip", 1.0, "grad norm clip (paper: 1.0)")
    flags.DEFINE_float("train_weight_decay", 1e-4, "weight decay (paper: 1e-4)")

    # paper total steps ~ 1e6
    flags.DEFINE_integer("epochs", 100, "epochs (epochs*steps_per_epoch ~= 1e6)")
    flags.DEFINE_integer("steps_per_epoch", 10_000, "steps per epoch")
    flags.DEFINE_integer("loss_freq", 1000, "loss print frequency (steps)")
    flags.DEFINE_integer("plot_freq", 10_000, "plot frequency (steps)")
    flags.DEFINE_integer("time_freq", 1000, "time estimate frequency (steps)")
    flags.DEFINE_integer("list_size", 0, "optional list size to increase effective batch (0=off)")
    flags.DEFINE_integer("plot_num", None, "number of plot cases")

    app.run(main)
