//! Safe, idiomatic Rust bindings for Apple's [MLX](https://github.com/ml-explore/mlx)
//! array framework, built on top of the [`mlxcore-sys`] FFI layer.
//!
//! This crate is Apple Silicon (macOS) only.
//!
//! # Suffix float literals with `f32`
//!
//! Unsuffixed float literals are `f64` in Rust, and Apple GPUs have no float64
//! support. Since the default stream is the GPU, `&[1.0, 2.0]` builds an array
//! that fails on every operation. Write `&[1.0f32, 2.0]`, and `&a * 2.0f32` for
//! scalar operands (MLX promotes to the wider dtype, so an unsuffixed literal
//! widens the result and the operator panics).
//!
//! float64 still works on an explicit [`Stream::cpu`] stream.
//!
//! # Threading: use one thread
//!
//! [`Array`] and [`Stream`] are deliberately **not** `Send` and not `Sync`, so
//! the compiler keeps all MLX work on the thread that started it. Two reasons,
//! and the first is the interesting one:
//!
//! - MLX is lazy, so an array owns the *graph* that produces it, and evaluating
//!   one array mutates the nodes it was built from — clearing their inputs and
//!   flipping their status, with no lock. Those nodes belong to the arrays you
//!   built from, which Rust sees as separate owned values. `Array` is `Rc`-like
//!   in this one respect, so even moving an array to another thread while its
//!   operands stay here would race: `Send` alone would not be sound.
//! - MLX's Metal backend keeps one command encoder per stream, and two threads
//!   submitting GPU work at once trips an assertion inside Metal.
//!
//! To use results elsewhere, copy them out first: [`Array::to_vec`] and
//! [`Array::item`] return plain Rust values, which are `Send` as usual.

mod array;
mod dtype;
mod error;
mod ffi;
mod stream;

pub mod random;

pub use array::Array;
pub use dtype::{ArrayElement, Dtype};
pub use error::{Error, Result};
pub use stream::Stream;

/// Returns the version string of the underlying MLX library.
pub fn version() -> String {
    use std::ffi::CStr;
    // SAFETY: standard mlx-c string-handle dance; all handles are freed.
    unsafe {
        let mut s = mlxcore_sys::mlx_string_new();
        mlxcore_sys::mlx_version(&mut s);
        let v = CStr::from_ptr(mlxcore_sys::mlx_string_data(s))
            .to_string_lossy()
            .into_owned();
        mlxcore_sys::mlx_string_free(s);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_version() {
        assert!(!version().is_empty());
    }

    /// Asks whether `T` is `Send`/`Sync` without requiring that it is.
    ///
    /// Rust has no `where T: !Send`, so this leans on method resolution: the
    /// inherent methods below exist only when the bound holds, and inherent
    /// methods win over trait ones. When the bound does not hold the inherent
    /// candidate is discarded and resolution falls through to the blanket trait
    /// impl, which answers `false`.
    struct Probe<T>(std::marker::PhantomData<T>);

    trait MaybeSend {
        fn is_send(&self) -> bool {
            false
        }
    }
    impl<T> MaybeSend for Probe<T> {}
    impl<T: Send> Probe<T> {
        fn is_send(&self) -> bool {
            true
        }
    }

    trait MaybeSync {
        fn is_sync(&self) -> bool {
            false
        }
    }
    impl<T> MaybeSync for Probe<T> {}
    impl<T: Sync> Probe<T> {
        fn is_sync(&self) -> bool {
            true
        }
    }

    fn probe<T>() -> Probe<T> {
        Probe(std::marker::PhantomData)
    }

    #[test]
    fn probe_detects_a_thread_safe_type() {
        // Guards the two tests below: a probe that always answered `false` would
        // let them pass no matter what `Array` and `Stream` implement.
        assert!(probe::<Vec<u8>>().is_send());
        assert!(probe::<Vec<u8>>().is_sync());
        // And the negative direction, on a type known to be neither.
        assert!(!probe::<std::rc::Rc<u8>>().is_send());
        assert!(!probe::<std::rc::Rc<u8>>().is_sync());
    }

    #[test]
    fn array_is_thread_bound() {
        // Deliberate: evaluating an array mutates graph nodes its operands share,
        // so neither moving nor sharing one across threads is sound. See the
        // crate docs before changing this.
        assert!(!probe::<Array>().is_send());
        assert!(!probe::<Array>().is_sync());
    }

    #[test]
    fn stream_is_thread_bound() {
        assert!(!probe::<Stream>().is_send());
        assert!(!probe::<Stream>().is_sync());
    }
}
