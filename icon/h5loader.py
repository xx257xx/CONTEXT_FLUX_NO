# h5loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Optional, Tuple, Sequence, List, Union

import glob
import numpy as np
import h5py

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import haiku as hk
from einshape import jax_einshape as einshape


# =========================
# Low-level utilities
# =========================

@dataclass
class OperatorRecord:
    equation: str
    coeffs: Dict[str, float]
    cond_k: np.ndarray  # (P, Nx, 1)
    cond_v: np.ndarray  # (P, Nx, 1)
    qoi_k: np.ndarray   # (P, Nx, 1)
    qoi_v: np.ndarray   # (P, Nx, 1)


def _find_first(f: h5py.File, candidates) -> Optional[h5py.Dataset]:
    """HDF5 안에서 candidate 이름들 중 첫 번째로 존재하는 dataset 반환"""
    for name in candidates:
        if name in f and isinstance(f[name], h5py.Dataset):
            return f[name]
    return None


def _infer_dt_stride(t: np.ndarray, tau: float) -> Tuple[float, int]:
    dt = float(t[1] - t[0])
    stride = int(round(tau / dt))
    if not np.isclose(stride * dt, tau, rtol=0, atol=1e-10):
        raise ValueError(
            f"tau/dt must be integer. dt={dt}, tau={tau}, stride={stride}, stride*dt={stride*dt}"
        )
    return dt, stride


def _ensure_3d_pair(arr: np.ndarray, dtype=np.float32) -> np.ndarray:
    """(P,Nx) -> (P,Nx,1), (P,Nx,1) 유지"""
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D pair array, got shape={arr.shape}")
    return arr


def _pad_trunc_2d(x: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: (N, D)
    returns:
      y: (L, D)  padded/truncated
      mask: (L,) bool
    """
    N, D = x.shape
    if N >= L:
        return x[:L], np.ones((L,), dtype=bool)
    y = np.zeros((L, D), dtype=x.dtype)
    y[:N] = x
    mask = np.zeros((L,), dtype=bool)
    mask[:N] = True
    return y, mask


def _one_hot(idx: int, K: int) -> np.ndarray:
    v = np.zeros((K,), dtype=np.float32)
    v[idx] = 1.0
    return v


def _build_prompt_and_mask_numpy(
    demo_cond_k: np.ndarray,  # (demo_num, cond_len, k_dim)
    demo_cond_v: np.ndarray,  # (demo_num, cond_len, v_dim)
    demo_qoi_k: np.ndarray,   # (demo_num, qoi_len, k_dim)
    demo_qoi_v: np.ndarray,   # (demo_num, qoi_len, v_dim)
    quest_cond_k: np.ndarray, # (1, cond_len, k_dim)
    quest_cond_v: np.ndarray, # (1, cond_len, v_dim)
    demo_num: int,
    cond_len: int,
    qoi_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TFRecord DataProvider의 build_prompt_and_mask 로직을 numpy로 재구현.
    prompt_dim = k_dim + v_dim + (demo_num+1)
    prompt_len = demo_num*(cond_len+qoi_len) + cond_len
    """
    index_dim = demo_num + 1
    prompt_list = []
    mask_list = []

    # demo blocks
    for i in range(demo_num):
        cond_index = np.tile(_one_hot(i, index_dim)[None, :], (cond_len, 1))      # (cond_len, demo_num+1)
        qoi_index  = -np.tile(_one_hot(i, index_dim)[None, :], (qoi_len, 1))     # (qoi_len, demo_num+1)

        demo_cond_i = np.concatenate([demo_cond_k[i], demo_cond_v[i], cond_index], axis=-1)
        demo_qoi_i  = np.concatenate([demo_qoi_k[i],  demo_qoi_v[i],  qoi_index],  axis=-1)

        prompt_list.append(demo_cond_i)
        prompt_list.append(demo_qoi_i)

        mask_list.append(np.ones((cond_len,), dtype=bool))
        mask_list.append(np.ones((qoi_len,), dtype=bool))

    # quest condition block
    quest_index = np.tile(_one_hot(demo_num, index_dim)[None, :], (cond_len, 1))
    quest_cond = np.concatenate([quest_cond_k[0], quest_cond_v[0], quest_index], axis=-1)

    prompt_list.append(quest_cond)
    mask_list.append(np.ones((cond_len,), dtype=bool))

    prompt = np.concatenate(prompt_list, axis=0).astype(np.float32)  # (prompt_len, prompt_dim)
    prompt_mask = np.concatenate(mask_list, axis=0)                  # (prompt_len,)

    # apply mask to prompt (명시적으로)
    prompt = prompt * prompt_mask.astype(np.float32)[:, None]
    return prompt, prompt_mask


# =========================
# Optional: keep your iterator (unchanged API)
# =========================

def iter_cubic_records_from_hdf5(
    path: str,
    *,
    tau: float = 0.1,
    t_init_window: float = 0.4,
    pairs_per_operator: int = 10_000,
    seed: int = 0,
    problem_type: Literal["forward", "backward"] = "forward",
    dtype=np.float32,
    equation_prefix: str = "conservation_cubic",
    fallback_t: Optional[np.ndarray] = None,
) -> Iterator[OperatorRecord]:
    """(너가 준 코드 그대로 유지)"""
    with h5py.File(path, "r") as f:
        ds_cond_v = _find_first(f, ["cond_v", "cond/value", "cond_values", "condV"])
        ds_qoi_v  = _find_first(f, ["qoi_v", "qoi/value", "qoi_values", "qoiV"])
        ds_cond_k = _find_first(f, ["cond_k", "cond/key", "cond_keys", "condK"])
        ds_qoi_k  = _find_first(f, ["qoi_k", "qoi/key", "qoi_keys", "qoiK"])

        ds_values = _find_first(f, ["values", "u", "U", "solution", "solutions"])
        ds_coeffs = _find_first(f, ["coeffs", "coeff", "params", "parameters"])
        ds_t = _find_first(f, ["t", "time", "times"])
        ds_x = _find_first(f, ["x", "x_grid", "grid", "space"])

        if ds_coeffs is None:
            raise KeyError("HDF5에서 coeffs/params를 찾지 못했습니다. inspect_hdf5로 키 이름을 확인해 주세요.")

        coeffs_all = np.asarray(ds_coeffs[...])
        if coeffs_all.ndim == 1 and coeffs_all.size == 3:
            coeffs_all = coeffs_all[None, :]
        if coeffs_all.shape[-1] < 3:
            raise ValueError(f"coeffs shape이 (n_pde,3) 형태가 아닙니다: {coeffs_all.shape}")

        n_pde = coeffs_all.shape[0]

        if ds_x is not None:
            x = np.asarray(ds_x[...])
        else:
            if ds_cond_k is None:
                raise KeyError("HDF5에서 x를 찾지 못했고 cond_k도 없습니다.")
            x = None

        if ds_t is not None:
            t = np.asarray(ds_t[...])
        else:
            t = fallback_t

        if (ds_cond_v is None or ds_qoi_v is None) and (t is None):
            raise KeyError("HDF5에 t가 없고 fallback_t도 None이라 full trajectory에서 pair를 만들 수 없습니다.")

        # Case A: pairs already present
        if ds_cond_v is not None and ds_qoi_v is not None:
            for p in range(n_pde):
                cond_v = _ensure_3d_pair(ds_cond_v[p], dtype=dtype)
                qoi_v  = _ensure_3d_pair(ds_qoi_v[p],  dtype=dtype)

                if ds_cond_k is not None and ds_qoi_k is not None:
                    cond_k = _ensure_3d_pair(ds_cond_k[p], dtype=dtype)
                    qoi_k  = _ensure_3d_pair(ds_qoi_k[p],  dtype=dtype)
                else:
                    if x is None:
                        raise KeyError("cond_k/qoi_k가 없고 x도 없어서 key를 만들 수 없습니다.")
                    x_key = x.astype(dtype)[:, None]
                    cond_k = np.broadcast_to(x_key[None, :, :], cond_v.shape).copy()
                    qoi_k  = np.broadcast_to(x_key[None, :, :], qoi_v.shape).copy()

                if problem_type == "backward":
                    cond_k, qoi_k = qoi_k, cond_k
                    cond_v, qoi_v = qoi_v, cond_v

                a, b, c = map(float, coeffs_all[p][:3])
                eqn = f"{equation_prefix}_{problem_type}_a={a:.8f}_b={b:.8f}_c={c:.8f}_tau={tau:g}"

                yield OperatorRecord(
                    equation=eqn,
                    coeffs={"a": a, "b": b, "c": c},
                    cond_k=cond_k,
                    cond_v=cond_v,
                    qoi_k=qoi_k,
                    qoi_v=qoi_v,
                )
            return

        # Case B: trajectory -> build pairs
        if ds_values is None:
            raise KeyError("cond_v/qoi_v도 없고 values도 없습니다.")

        if t is None:
            raise ValueError("full trajectory에서 pair를 만들려면 t가 필요합니다.")

        dt, stride = _infer_dt_stride(t, tau)
        max_start_idx = int(round(t_init_window / dt))
        start_idx = np.arange(max_start_idx + 1, dtype=np.int32)

        if x is None:
            raise KeyError("full trajectory에서 pair 생성하려면 x가 필요합니다.")

        x_key = x.astype(dtype)[:, None]

        for p in range(n_pde):
            u = np.asarray(ds_values[p], dtype=dtype)
            if u.ndim == 4 and u.shape[2] == 1:
                u = u[:, :, 0, :]
            if u.ndim != 3:
                raise ValueError(f"values[p] expected 3D (ic,t,x) after squeeze, got {u.shape}")

            n_ic, n_t, n_x = u.shape
            if start_idx[-1] + stride >= n_t:
                raise ValueError("Not enough time steps in values")

            cond_all = u[:, start_idx, :]
            qoi_all  = u[:, start_idx + stride, :]

            cond_flat = cond_all.reshape(-1, n_x)
            qoi_flat  = qoi_all.reshape(-1, n_x)

            total_pairs = cond_flat.shape[0]
            take = min(pairs_per_operator, total_pairs)

            rng = np.random.default_rng(seed + p)
            choose = rng.choice(total_pairs, size=take, replace=False)

            cond = cond_flat[choose][:, :, None]
            qoi  = qoi_flat[choose][:, :, None]

            if problem_type == "forward":
                cond_v, qoi_v = cond, qoi
            else:
                cond_v, qoi_v = qoi, cond

            cond_k = np.broadcast_to(x_key[None, :, :], cond_v.shape).copy()
            qoi_k  = np.broadcast_to(x_key[None, :, :], qoi_v.shape).copy()

            a, b, c = map(float, coeffs_all[p][:3])
            eqn = f"{equation_prefix}_{problem_type}_a={a:.8f}_b={b:.8f}_c={c:.8f}_tau={tau:g}"

            yield OperatorRecord(
                equation=eqn,
                coeffs={"a": a, "b": b, "c": c},
                cond_k=cond_k,
                cond_v=cond_v,
                qoi_k=qoi_k,
                qoi_v=qoi_v,
            )


# =========================
# HDF5 DataProvider (the missing piece)
# =========================

class _H5Bank:
    """
    한 개(또는 여러 개) HDF5 파일을 열어두고, 샘플링에 필요한 dataset 핸들을 잡고 있는 클래스.
    - pairs(cond_v/qoi_v)가 있으면 그걸 사용
    - 없으면 trajectory(values)에서 on-the-fly로 (u(t),u(t+tau)) pair 샘플링
    """
    def __init__(
        self,
        paths: Sequence[str],
        *,
        tau: float,
        t_init_window: float,
        dtype=np.float32,
        equation_prefix: str = "conservation_cubic",
    ):
        self.dtype = dtype
        self.tau = float(tau)
        self.t_init_window = float(t_init_window)
        self.equation_prefix = equation_prefix

        self.files: List[h5py.File] = []
        self.meta = []  # per file dict

        for p in paths:
            f = h5py.File(p, "r")
            self.files.append(f)

            ds_cond_v = _find_first(f, ["cond_v", "cond/value", "cond_values", "condV"])
            ds_qoi_v  = _find_first(f, ["qoi_v", "qoi/value", "qoi_values", "qoiV"])
            ds_cond_k = _find_first(f, ["cond_k", "cond/key", "cond_keys", "condK"])
            ds_qoi_k  = _find_first(f, ["qoi_k", "qoi/key", "qoi_keys", "qoiK"])

            ds_values = _find_first(f, ["values", "u", "U", "solution", "solutions"])
            ds_coeffs = _find_first(f, ["coeffs", "coeff", "params", "parameters"])
            ds_t = _find_first(f, ["t", "time", "times"])
            ds_x = _find_first(f, ["x", "x_grid", "grid", "space"])

            if ds_coeffs is None:
                raise KeyError(f"[{p}] coeffs/params dataset not found")

            coeffs_all = np.asarray(ds_coeffs[...])
            if coeffs_all.ndim == 1 and coeffs_all.size == 3:
                coeffs_all = coeffs_all[None, :]
            n_pde = int(coeffs_all.shape[0])

            x = np.asarray(ds_x[...]) if ds_x is not None else None
            t = np.asarray(ds_t[...]) if ds_t is not None else None

            mode = "pairs" if (ds_cond_v is not None and ds_qoi_v is not None) else "traj"
            if mode == "traj":
                if ds_values is None:
                    raise KeyError(f"[{p}] neither (cond_v,qoi_v) nor values found")
                if t is None:
                    raise KeyError(f"[{p}] trajectory mode requires t dataset")
                if x is None:
                    raise KeyError(f"[{p}] trajectory mode requires x dataset")
                dt, stride = _infer_dt_stride(t, self.tau)
                max_start_idx = int(round(self.t_init_window / dt))
                start_idx = np.arange(max_start_idx + 1, dtype=np.int32)
            else:
                dt = stride = None
                start_idx = None

            self.meta.append(
                dict(
                    path=p,
                    mode=mode,
                    n_pde=n_pde,
                    coeffs_all=coeffs_all,
                    x=x,
                    t=t,
                    dt=dt,
                    stride=stride,
                    start_idx=start_idx,
                    ds_cond_v=ds_cond_v,
                    ds_qoi_v=ds_qoi_v,
                    ds_cond_k=ds_cond_k,
                    ds_qoi_k=ds_qoi_k,
                    ds_values=ds_values,
                )
            )

        # global index map: choose file then pde
        self._file_offsets = np.cumsum([0] + [m["n_pde"] for m in self.meta])
        self.n_total_pde = int(self._file_offsets[-1])

    def close(self):
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass

    def _locate(self, global_pde_idx: int) -> Tuple[int, int]:
        # returns (file_idx, local_pde_idx)
        if global_pde_idx < 0 or global_pde_idx >= self.n_total_pde:
            raise IndexError(global_pde_idx)
        file_idx = int(np.searchsorted(self._file_offsets, global_pde_idx, side="right") - 1)
        local_idx = int(global_pde_idx - self._file_offsets[file_idx])
        return file_idx, local_idx

    def sample_pairs_for_operator(
        self,
        rng: np.random.Generator,
        global_pde_idx: int,
        n_pairs: int,
        direction: Literal["forward", "backward"] = "forward",
        select: Literal["random", "sequential"] = "sequential",
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        returns:
          equation(str),
          cond_k: (n_pairs, Nx, 1)
          cond_v: (n_pairs, Nx, 1)
          qoi_k : (n_pairs, Nx, 1)
          qoi_v : (n_pairs, Nx, 1)
        """
        fi, pi = self._locate(global_pde_idx)
        m = self.meta[fi]

        a, b, c = map(float, m["coeffs_all"][pi][:3])
        eqn = f"{self.equation_prefix}_{direction}_a={a:.8f}_b={b:.8f}_c={c:.8f}_tau={self.tau:g}"

        if m["mode"] == "pairs":
            ds_cond_v = m["ds_cond_v"]
            ds_qoi_v  = m["ds_qoi_v"]
            assert ds_cond_v is not None and ds_qoi_v is not None

            P = int(ds_cond_v.shape[1]) if ds_cond_v.ndim >= 2 else int(ds_cond_v.shape[0])
     
            if select == "sequential":
                idx = np.arange(n_pairs) % P
            else:
                idx = rng.integers(0, P, size=(n_pairs,), endpoint=False)

            # (n_pairs, Nx, 1) or (n_pairs, Nx)
            cond_v = _ensure_3d_pair(ds_cond_v[pi, idx], dtype=self.dtype)
            qoi_v  = _ensure_3d_pair(ds_qoi_v[pi, idx],  dtype=self.dtype)

            if m["ds_cond_k"] is not None and m["ds_qoi_k"] is not None:
                cond_k = _ensure_3d_pair(m["ds_cond_k"][pi, idx], dtype=self.dtype)
                qoi_k  = _ensure_3d_pair(m["ds_qoi_k"][pi, idx],  dtype=self.dtype)
            else:
                x = m["x"]
                if x is None:
                    raise KeyError("pairs mode: need either (cond_k,qoi_k) or x")
                x_key = x.astype(self.dtype)[:, None]  # (Nx,1)
                cond_k = np.broadcast_to(x_key[None, :, :], cond_v.shape).copy()
                qoi_k  = np.broadcast_to(x_key[None, :, :], qoi_v.shape).copy()

        else:
            # trajectory mode: sample (ic, start_idx) on the fly
            ds_values = m["ds_values"]
            assert ds_values is not None
            x = m["x"]
            t = m["t"]
            stride = int(m["stride"])
            start_idx = m["start_idx"]
            if x is None or t is None or start_idx is None:
                raise RuntimeError("trajectory meta missing")
           
            # u shape could be (ic, t, x) or (ic, t, 1, x)
            # we will gather pointwise for each sampled pair
            # Determine n_ic, n_t from ds_values[pi] without loading all
            u_shape = ds_values.shape  # maybe (n_pde,n_ic,n_t,n_x) or (n_pde,n_ic,n_t,1,n_x)
            if len(u_shape) == 4:
                _, n_ic, n_t, n_x = u_shape
                has_dim = False
            elif len(u_shape) == 5:
                _, n_ic, n_t, dim1, n_x = u_shape
                if dim1 != 1:
                    raise ValueError(f"expected dim==1, got {dim1}")
                has_dim = True
            else:
                raise ValueError(f"unexpected values shape: {u_shape}")

            ic_idx = rng.integers(0, n_ic, size=(n_pairs,), endpoint=False)

            if select == "sequential":
                startnum = rng.integers(0, len(start_idx)-200)
                si = np.arange(startnum, startnum + 10*n_pairs,10) 
                #print(si)
            else:       
                si = rng.integers(0, len(start_idx), size=(n_pairs,), endpoint=False)
                

            t0 = start_idx[si]
            t1 = t0 + stride
            #print(t0,t1)
            # build arrays
            cond_v = np.zeros((n_pairs, n_x, 1), dtype=self.dtype)
            qoi_v  = np.zeros((n_pairs, n_x, 1), dtype=self.dtype)
            #print(ic_idx,t0)
            #print(ds_values.shape  )
            for i in range(n_pairs):
                if not has_dim:
                    #print(t0[i])
                    cond_v[i, :, 0] = ds_values[pi, ic_idx[0], t0[i], :]
                    qoi_v[i, :, 0]  = ds_values[pi, ic_idx[0], t1[i], :]
                else:
                    #print(t0[i])
                    cond_v[i, :, 0] = ds_values[pi, ic_idx[0], t0[i], 0, :]
                    qoi_v[i, :, 0]  = ds_values[pi, ic_idx[0], t1[i], 0, :]

            x_key = x.astype(self.dtype)[:, None]  # (Nx,1)
            cond_k = np.broadcast_to(x_key[None, :, :], cond_v.shape).copy()
            qoi_k  = np.broadcast_to(x_key[None, :, :], qoi_v.shape).copy()

        # forward/backward swap
        if direction == "backward":
            cond_k, qoi_k = qoi_k, cond_k
            cond_v, qoi_v = qoi_v, cond_v

        # 기존: return eqn, cond_k, cond_v, qoi_k, qoi_v, si[-1]
        return eqn, cond_k, cond_v, qoi_k, qoi_v, ic_idx, si


class DataProvider:
    """
    run_h5py.py가 기대하는 인터페이스를 그대로 맞춘 HDF5용 DataProvider.

    get_next_data() -> (equation(list[str]), prompt, mask, query, query_mask, ground_truth)
    그리고 내부에서 multi-device용으로 (num_devices, batch_per_device, ...) reshape까지 수행.
    """
    def __init__(
        self,
        seed: int,
        demo_num: int,
        cond_len: int,
        qoi_len: int,
        batch_size: int,
        shuffle_buffer_size: int,
        file_names: Union[str, Sequence[str]],
        k_dim: int,
        v_dim: int,
        config: Optional[dict] = None,
        select: str = "sequential",
        k_mode: str = "naive",
        deterministic: bool = True,
        return_raw: bool = False,
        drop_remainder: bool = True,
        shuffle_dataset: bool = True,
        num_epochs=None,
        num_devices: int = len(jax.devices()),
        name: str = "DataProvider",
        # paper-ish controls
        tau: float = 0.1,
        t_init_window: float = 0.4,
        pairs_per_operator: int = 10_000,  # (pairs mode에서만 의미가 큼)
        direction: Literal["forward", "backward", "both"] = "forward",
        equation_prefix: str = "conservation_cubic",
        dtype=np.float32,
    ):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.hk_rng = hk.PRNGSequence(jax.random.PRNGKey(self.seed))
        self.name = name

        self.demo_num = int(demo_num)
        self.cond_len = int(cond_len)
        self.qoi_len = int(qoi_len)
        self.batch_size = int(batch_size)

        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim)
        self.k_mode = str(k_mode)
        self.select = str(select)

        self.num_devices = int(num_devices)
        if self.batch_size % self.num_devices != 0:
            raise ValueError(f"batch_size({self.batch_size}) must be divisible by num_devices({self.num_devices})")

        self.tau = float(tau)
        self.t_init_window = float(t_init_window)
        self.pairs_per_operator = int(pairs_per_operator)
        self.direction = direction
        self.dtype = dtype

        # expand globs
        if isinstance(file_names, str):
            names = [file_names]
        else:
            names = list(file_names)

        expanded: List[str] = []
        for n in names:
            if "*" in n or "?" in n or "[" in n:
                expanded += sorted(glob.glob(n))
            else:
                expanded.append(n)
        if len(expanded) == 0:
            raise FileNotFoundError(f"No files matched: {file_names}")

        self.bank = _H5Bank(
            expanded,
            tau=self.tau,
            t_init_window=self.t_init_window,
            dtype=self.dtype,
            equation_prefix=equation_prefix,
        )

        # for printing/debug
        self._last_raw = None

    def __del__(self):
        try:
            self.bank.close()
        except Exception:
            pass

    def _sample_direction_for_batch(self) -> List[Literal["forward", "backward"]]:
        if self.direction == "forward":
            return ["forward"] * self.batch_size
        if self.direction == "backward":
            return ["backward"] * self.batch_size
        # both: half forward, half backward (paper mix)
        half = self.batch_size // 2
        dirs = ["forward"] * half + ["backward"] * (self.batch_size - half)
        # shuffle within batch for diversity
        self.rng.shuffle(dirs)
        return dirs

    def get_next_data(
        self,
        decode_equation: bool = False,  # kept for compatibility; equation already str
        list_size: int = 0,
        return_raw: bool = False,
    ):
        if list_size != 0:
            raise NotImplementedError("HDF5 DataProvider: list_size>0 not implemented (use 0)")

        dirs = self._sample_direction_for_batch()

        prompt_list = []
        mask_list = []
        query_list = []
        query_mask_list = []
        gt_list = []
        eqn_list: List[str] = []
        raw_list = []
        si_list = []
        ic_list = []

        pde_list = []
        for bi in range(self.batch_size):
            # choose random operator among all PDEs
            pde_idx = int(self.rng.integers(0, self.bank.n_total_pde))

            # we need (demo_num demos) + (1 quest)
            n_pairs = self.demo_num + 1

            eqn, cond_k, cond_v, qoi_k, qoi_v, ic_idx, si = self.bank.sample_pairs_for_operator(
                self.rng, pde_idx, n_pairs=n_pairs, direction=dirs[bi], select=self.select
            )

            # split demos vs quest
            demo_cond_k = cond_k[: self.demo_num]  # (demo_num, Nx, 1)
            demo_cond_v = cond_v[: self.demo_num]
            demo_qoi_k  = qoi_k[: self.demo_num]
            demo_qoi_v  = qoi_v[: self.demo_num]

            quest_cond_k = cond_k[self.demo_num : self.demo_num + 1]  # (1, Nx, 1)
            quest_cond_v = cond_v[self.demo_num : self.demo_num + 1]
            quest_qoi_k  = qoi_k[self.demo_num : self.demo_num + 1]
            quest_qoi_v  = qoi_v[self.demo_num : self.demo_num + 1]

            # (Nx,1) -> (cond_len,k_dim)/(cond_len,v_dim) with pad/trunc
            # keys
            def prep_k(x_3d: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
                # x_3d: (B, Nx, 1)
                B, Nx, _ = x_3d.shape
                out = np.zeros((B, L, self.k_dim), dtype=np.float32)
                mask = np.zeros((B, L), dtype=bool)
                for i in range(B):
                    x2 = x_3d[i].astype(np.float32)  # (Nx,1)
                    x2, m2 = _pad_trunc_2d(x2, L)    # (L,1), (L,)
                    # pad to k_dim
                    if self.k_dim > x2.shape[1]:
                        x2 = np.pad(x2, ((0, 0), (0, self.k_dim - x2.shape[1])))
                    out[i] = x2[:, : self.k_dim]
                    mask[i] = m2
                return out, mask

            # values
            def prep_v(x_3d: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
                B, Nx, _ = x_3d.shape
                out = np.zeros((B, L, self.v_dim), dtype=np.float32)
                mask = np.zeros((B, L), dtype=bool)
                for i in range(B):
                    x2 = x_3d[i].astype(np.float32)  # (Nx,1)
                    x2, m2 = _pad_trunc_2d(x2, L)
                    if self.v_dim > x2.shape[1]:
                        x2 = np.pad(x2, ((0, 0), (0, self.v_dim - x2.shape[1])))
                    out[i] = x2[:, : self.v_dim]
                    mask[i] = m2
                return out, mask

            demo_cond_k2, demo_cond_mask = prep_k(demo_cond_k, self.cond_len)
            demo_cond_v2, _             = prep_v(demo_cond_v, self.cond_len)
            demo_qoi_k2,  demo_qoi_mask = prep_k(demo_qoi_k,  self.qoi_len)
            demo_qoi_v2,  _             = prep_v(demo_qoi_v,  self.qoi_len)

            quest_cond_k2, quest_cond_mask = prep_k(quest_cond_k, self.cond_len)
            quest_cond_v2, _               = prep_v(quest_cond_v, self.cond_len)
            quest_qoi_k2,  quest_qoi_mask  = prep_k(quest_qoi_k,  self.qoi_len)
            quest_qoi_v2,  _               = prep_v(quest_qoi_v,  self.qoi_len)

            # prompt/mask
            prompt, pmask = _build_prompt_and_mask_numpy(
                demo_cond_k2, demo_cond_v2, demo_qoi_k2, demo_qoi_v2,
                quest_cond_k2, quest_cond_v2,
                demo_num=self.demo_num,
                cond_len=self.cond_len,
                qoi_len=self.qoi_len,
            )

            # query/gt/mask (quest qoi)
            query = quest_qoi_k2[0].astype(np.float32)         # (qoi_len, k_dim)
            gt    = quest_qoi_v2[0].astype(np.float32)         # (qoi_len, v_dim)
            qmask = quest_qoi_mask[0].astype(bool)             # (qoi_len,)
        
            ic_list.append(ic_idx)   # ic_idx shape: (n_pairs,)a
            
            prompt_list.append(prompt)
            mask_list.append(pmask)
            query_list.append(query)
            gt_list.append(gt)
            query_mask_list.append(qmask)
            eqn_list.append(eqn)
            si_list.append(si)
            pde_list.append(pde_idx)

            if return_raw:
                raw_list.append((cond_k, cond_v, qoi_k, qoi_v))


        # stack batch
        
        pde_b = np.array(pde_list, dtype=np.int32)
        prompt_b = np.stack(prompt_list, axis=0)         # (B, prompt_len, prompt_dim)
        mask_b   = np.stack(mask_list, axis=0)           # (B, prompt_len)
        query_b  = np.stack(query_list, axis=0)          # (B, qoi_len, k_dim)
        qmask_b  = np.stack(query_mask_list, axis=0)     # (B, qoi_len)
        gt_b     = np.stack(gt_list, axis=0)             # (B, qoi_len, v_dim)

        # reshape to (num_devices, batch_per_device, ...)
        B = self.batch_size
        nd = self.num_devices
        bd = B // nd

        prompt_b = prompt_b.reshape(nd, bd, *prompt_b.shape[1:])
        mask_b   = mask_b.reshape(nd, bd, *mask_b.shape[1:])
        query_b  = query_b.reshape(nd, bd, *query_b.shape[1:])
        qmask_b  = qmask_b.reshape(nd, bd, *qmask_b.shape[1:])
        gt_b     = gt_b.reshape(nd, bd, *gt_b.shape[1:])

        # to jax arrays
        prompt_b = jnp.asarray(prompt_b)
        mask_b   = jnp.asarray(mask_b)
        query_b  = jnp.asarray(query_b)
        qmask_b  = jnp.asarray(qmask_b)
        gt_b     = jnp.asarray(gt_b)


        pde_b = np.array(pde_list, dtype=np.int32)        # (B,)
        ic_b  = np.stack(ic_list, axis=0).astype(np.int32) # (B, n_pairs)
        si_b  = np.stack(si_list, axis=0).astype(np.int32) # (B, n_pairs)  (or (B,) if you stored scalar)

        # reshape to (nd, bd, ...)
        pde_b = pde_b.reshape(nd, bd)
        ic_b  = ic_b.reshape(nd, bd, *ic_b.shape[1:])
        si_b  = si_b.reshape(nd, bd, *si_b.shape[1:])

        # to jax
        pde_b = jnp.asarray(pde_b)
        ic_b  = jnp.asarray(ic_b)
        si_b  = jnp.asarray(si_b)
        if return_raw:
            return raw_list,eqn_list, prompt_b, mask_b, query_b, qmask_b, gt_b, pde_b, ic_b, si_b
        return eqn_list, prompt_b, mask_b, query_b, qmask_b, gt_b, si_b

    def pretty_print(self, equation, prompt, mask, query, query_mask, ground_truth):
        from pprint import pprint
        pprint(equation[: min(5, len(equation))])
        print("prompt size:", tree.tree_map(lambda x: x.shape, prompt), flush=True)
        print("mask size:", tree.tree_map(lambda x: x.shape, mask), flush=True)
        print("query size:", tree.tree_map(lambda x: x.shape, query), flush=True)
        print("query_mask size:", tree.tree_map(lambda x: x.shape, query_mask), flush=True)
        print("ground_truth size:", tree.tree_map(lambda x: x.shape, ground_truth), flush=True)


# =========================
# Inspect helper
# =========================

def inspect_hdf5(path: str, max_attrs: int = 20):
    def _print(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DS] {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GRP] {name}/")
    with h5py.File(path, "r") as f:
        print("=== FILE ATTRS ===")
        for i, (k, v) in enumerate(f.attrs.items()):
            if i >= max_attrs:
                print("... (attrs truncated)")
                break
            print(k, "=", v)
        print("\n=== TREE ===")
        f.visititems(_print)
