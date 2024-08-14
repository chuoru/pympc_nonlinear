# README 

A work-around for using optimization engine in Windows environment.
The problem with `extern crate` is only happens with Windows environment and optimization engine of version `0.9.0` when new constraints are included. The root of this problem might be the symlink of Windows compilers is not working as expected in Rust.   

To use this work-around, keep `Cargo-lock` and folder structure as `rust/navigation` and `rust/navigation/tcp_iface_navigation` as it is.

Modify your python code as below:
```
build_config = og.config.BuildConfiguration()\
    .with_build_directory("rust")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
```

```
mng = og.tcp.OptimizerTcpManager('rust/navigation')
```


Issue to track
- Optimization Engine: https://github.com/alphaville/optimization-engine/issues/348
- Rust: https://github.com/rust-lang/rust/issues/86125