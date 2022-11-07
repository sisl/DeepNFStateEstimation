#=
bicycle_model.jl:
    A file containing shared functions for kinematic bicycle model dynamics, 
    trajectory generation, and dataset creation
=#

##
#*******************************************************************************
# PACKAGES AND SETUP
#*******************************************************************************
using Distributions
using HDF5
# using PGFPlots
using Random

# # Color-blind color palette
# define_color("pastelRed",       [245,97,92])
# define_color("pastelBlue",      [0,114,178])
# define_color("pastelGreen",     [0,158,115])
# define_color("pastelPurple",    [135,112,254])

##
#*******************************************************************************
# BICYCLE MODEL DEFINITIONS
#*******************************************************************************
Δt = 0.1            # Discrete time step
tf = 15             # End time [s]
times = 0:Δt:tf     # Simulation times
viz_step = 15       # Step size between sampled trajectory points for plots
N = 1000            # Number of trajectories to simulate
n_traj = N
L = 2.8             # Bicycle model wheel base [m]

x0 = 0.0    # Initial value for x-coordinate [m]
y0 = 0.0    # Initial value for y-coordinate [m]
θ0 = 0.0    # Initial value for heading angle [rad]
v0 = 6.0    # Initial value for velocity [m/s]
ϕ0 = 0.0    # Initial value for steering angle [rad]           
t0 = 0.0    # Initial value for time [s]  

σv = 0.005  # Standard deviation for velocity [m/s]
σϕ = 0.001 # Standard deviation for steering angle [rad]

##
#*******************************************************************************
# FUNCTIONS
#*******************************************************************************
# Nominal transition function for kinematic bicycle model: 𝐱 = [x y θ v ϕ t]'
# Note: rear-axle reference model
function bicycle_dynamics(s; withNoise = false)
    x = s[1]; y = s[2]; θ = s[3]; v = s[4]; ϕ = s[5]; t = s[6]
    x′ = x + Δt*(v*cos(θ))
    y′ = y + Δt*(v*sin(θ))
    θ′ = θ + Δt*v*tan(ϕ)/L
    v′ = v + withNoise*rand(Normal(0.0, σv))
    ϕ′ = ϕ + Δt*0.1*0.5*cos(0.5*t) + withNoise*rand(Normal(0.0, σϕ))
    t′ = t + Δt*1
    s′ = [x′; y′; θ′; v′; ϕ′; t′]
    return s′
end

# Function to simulate N trajectories
function simulate_τ(N, s0, T, t; drop_rate = 0.0)
    τ_arr = []
    for j = 1:N
        τ = []
        push!(τ, s0)
        for i = 2:t
            push!(τ, T(τ[i-1]))
        end
        push!(τ_arr, τ[rand(t) .> drop_rate])
    end
    return τ_arr
end

# Function to store sample x,y data at the same time step across simulations
function extract_xy(τ_arr, viz_step)
    plot_x = []; plot_y = []
    for i = 1:lastindex(τ_arr)
        x = [s[1] for s in τ_arr[i]][1:viz_step:end]
        y = [s[2] for s in τ_arr[i]][1:viz_step:end]
        push!(plot_x, x); push!(plot_y, y)
    end
    return plot_x, plot_y
end

##
#*******************************************************************************
# ROLLOUT AND PLOT TRAJECTORIES
#*******************************************************************************
N = 250 # Number of trajectories to simulate
# Simulate trajectories with the noisy dynamics
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(N, s0, s->bicycle_dynamics(s, withNoise=true), t);

# Store sample x,y data to plot at the same time step across simulations
plot_x, plot_y = extract_xy(τ_arr, viz_step);

# # Define the axis
# p = Axis(style="enlarge x limits=false,grid=both, no marks", 
#                 xlabel="x", ylabel="y",title="Bicycle Model Rollouts",
#                 legendPos = "north east",legendStyle="nodes = {scale = 0.75}")

for i = 1:Int(ceil(length(times)/viz_step))
    # Extract the sampled points from the trajectory rollouts
    x_points = [cluster[i] for cluster in plot_x]
    y_points = [cluster[i] for cluster in plot_y]
    
    # # Plot the ellipses and sampled points
    # push!(p, PGFPlots.Linear(x_points, y_points, 
    #     style = "pastelBlue, only marks, mark options=
    #         {scale=0.25,fill=pastelBlue, solid, mark = o}, forget plot"))
end
# save("figs/bicycle_rollouts.pdf", p)

##
#*******************************************************************************
# DATASET GENERATION
#*******************************************************************************
N = 1000 # Number of trajectories to simulate
# Simulate trajectories with the noisy dynamics
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(N, s0, s->bicycle_dynamics(s, withNoise=true), t, drop_rate=0.9);

# Store the x, y, and time data
x = [s[1] for i in 1:lastindex(τ_arr) for s in τ_arr[i]]
y = [s[2] for i in 1:lastindex(τ_arr) for s in τ_arr[i]]
t = [s[6] for i in 1:lastindex(τ_arr) for s in τ_arr[i]]

# # Plot the stored data
# p = Axis(style="enlarge x limits=false,grid=both, no marks", 
#             xlabel="x", ylabel="y",title="Bicycle Model Rollouts",
#             legendPos = "north east",legendStyle="nodes = {scale = 0.75}")
# push!(p, PGFPlots.Linear(x, y, 
#             style = "pastelBlue, only marks, mark options=
#             {scale=0.25,fill=pastelBlue, solid, mark = o}, forget plot"))

# save("figs/bicycle_dataset.pdf", p)

# Save the data
fid = h5open("bicycle_dataset.h5", "w")
create_group(fid, "group")
g = fid["group"]
g["bicycle_data"] = [x y t]
close(fid)

##
# Save the data
fid = h5open("bicycle_dataset.h5", "w")
fid["position"] = [x y]
fid["time"] = t
#g = fid["group"]
#g["bicycle_data"] = [x y t]
close(fid)