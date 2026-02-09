# kinetic electrostatic particle in cell plasma simulation in 3D

This simulation models both electrons and ions in a 3d box.

This video shows a collection of electrons initially positioned in one corner and propagating through 
the box over a period of 2 microseconds (2e-6 seconds). 

All box faces are neutral, but reflect particles specularly.  The model 
includes charges and fields created by the charge of electrons as well as (unshown) oxygen
ions (+1).

https://github.com/user-attachments/assets/ec5d5c42-b0f0-42f0-a2f0-86e47150eefb

## Execution and debugging
There are two different tools used for debugging and assertion handling.

First, the usual Rust debug_assert! functionality is used to enable "typical"
assertions, and will be run by default in `cargo run` and `cargo test` executions.  
This can be turned off using --release, as in `cargo run --release`.

Second, more extensive checking is gated through a feature flag called "bounds-check". 
In order to enable this on the command line, use cargo run --features bounds-check.

Combined, these two allow for four different configurations for debugging, from slowest:

`cargo run --features bounds-check <binary-target>`

to a somewhat odd configuration (probably not so useful):

`cargo run --release --features bounds-check <binary-target>`

to a typical run during development:

`cargo run <binary-target>`

to fastest:

`cargo run --release <binary-target>`




