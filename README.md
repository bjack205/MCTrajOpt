# Maximal Coordinate Trajectory Optimization
This repo implements trajectory optimization for multi-body systems using maximal 
coordinates. The code interfaces with [`MathOptInterface`](https://jump.dev/MathOptInterface.jl/stable/) to provide support for the Ipopt solver. 

## Code Layout
All of the core code is located in the `src` folder. Right now the double-pendulum 
system is special-cased and separated into the `src/double_pendulum.jl` file and the 
`src/dp_ipopt.jl` files. Similarly, the code for a more generic serial-link arm system 
is located in `src/arm.jl` and `src/arm_ipopt.jl`. 

Other notable files:
* `src/rotations.jl` - all the code for dealing with rotations
* `src/joints.jl` - code for evaluating everything related to revolute joints
* `src/sparse.jl` - a helper class for filling in sparse Jacobians in Ipopt
* 'src/ipopt_helpers.jl` - a function for parsing Ipopt output into a dictionary

## Examples
The examples in the paper can be run from the scripts in the `examples` directory.

Output trajectories:

| Minimal coords | Maximal Coords | Arm |
| -------------  | -------------- | --- |
| <img src="https://github.com/bjack205/MCTrajOpt/blob/main/examples/acrobot_min.gif" alt="min coord" width="300"/> | <img src="https://github.com/bjack205/MCTrajOpt/blob/main/examples/acrobot_max.gif" alt="max coord" width="300"/> | <img src="https://github.com/bjack205/MCTrajOpt/blob/main/examples/arm.gif" alt="min coord" width="300"/>
