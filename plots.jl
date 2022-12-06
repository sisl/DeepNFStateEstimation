#=
plots.jl:
    A file containing functions for reading stored experimental data and
    generating plots of level sets and loss curves
=#

##
#*******************************************************************************
# PACKAGES AND SETUP
#*******************************************************************************
using CSV
using DataFrames
using Distributions
using HDF5
using LinearAlgebra
using PGFPlots

# Color-blind color palette
define_color("pastelRed",       [245,97,92])
define_color("pastelBlue",      [0,114,178])
define_color("pastelGreen",     [0,158,115])
define_color("pastelPurple",    [135,112,254])

define_color("viridis1",   [227,228,18])
define_color("viridis2",   [31,161,135])
define_color("viridis3",   [70,50,127])

define_color("cardinal",    [140,21,21])
define_color("fire_opal",       [223,96,71])
define_color("pastelRed",       [245,97,92])
define_color("dark_red", [220, 0, 0])

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

# Uncented transform
function unscented_transform(μ, Σ, λ, T, ws, n)
    # Enforce that the matrix is Hermitian
    Δ = cholesky(Hermitian((n + λ) * Σ)).L
    S = [μ]
    for i in 1:n
        push!(S, μ + Δ[:,i])
        push!(S, μ - Δ[:,i])
    end
    S′ = T.(S)
    μ′ = sum(w*s for (w,s) in zip(ws, S′))
    Σ′ = sum(w*(s - μ′)*(s - μ′)' for (w,s) in zip(ws, S′))
    return (μ′, Σ′, S′)
end

# Function to plot a conventional error ellipse
function plot_ellipse(μ, Σ, prob, color, legendentry = nothing)
    r = sqrt(-2*log(1-prob))
    angles = LinRange(0,2*pi, 200)
    x = [r*cos(α) for α in angles]; y = [r*sin(α) for α in angles]
    ellipse = Float64.(sqrt(Σ)) * [x'; y'] .+ μ'
    ex = ellipse[1,:]; ey = ellipse[2,:]
    if isnothing(legendentry)
        p = Plots.Linear(ex, ey, style=color*", thick, solid, forget plot")
    else
        p = Plots.Linear(
                ex, ey, style=color*", thick, solid", legendentry = legendentry)
    end
    return p
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


##
#*******************************************************************************
# UNSCENTED TRANSFORM
#*******************************************************************************
Δt = 0.1            # Discrete time step
tf = 15             # End time [s]
times = 0:Δt:tf     # Simulation times
viz_step = 20       # Step size between sampled trajectory points for plots
N = 1000            # Number of trajectories to simulate
L = 2.8             # Bicycle model wheel base [m]

ϵ = 1e-10           # Epsilon value for Cholesky matrix decomp

x0 = 0.0;           σx = ϵ      # Initial value and std for x-coordinate [m]
y0 = 0.0;           σy = ϵ      # Initial value and std for y-coordinate [m]
θ0 = 0.0;           σθ = ϵ      # Initial value and std for heading angle [rad]
v0 = 6.0;           σv = 0.01   # Initial value and std for velocity [m/s]
ϕ0 = 0.0;           σϕ = 0.0025 # Initial value and std for steering angle [rad]
t0 = 0.0;           σt = ϵ      # Initial value and std for time [rad]       

μ0 = [x0; y0; θ0; v0; ϕ0; t0]
Σ0 = zeros(6,6)
Σ0[diagind(Σ0)] = [σx^2, σy^2, σθ^2, σv^2, σϕ^2, σt^2]

λ = 2
n = length(μ0)
ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

# Store initial sigma points
Δ0 = cholesky(Hermitian((n + λ) * Σ0)).L
S0 = [μ0]
for i in 1:n
    push!(S0, μ0 + Δ0[:,i])
    push!(S0, μ0 - Δ0[:,i])
end

μp = μ0; Σp = Σ0
μ_ukf_arr = []; Σ_ukf_arr = []; S_arr = []
push!(μ_ukf_arr, μp); push!(Σ_ukf_arr, Σp); push!(S_arr, S0);
for i = 2:lastindex(times)
    # INDEX INTO SEQ HERE 
    μp, Σp, S = unscented_transform(μp, Σp, λ, s->bicycle_dynamics(s), ws, n)
    Σp = Σp + Σ0
    push!(μ_ukf_arr, μp); push!(Σ_ukf_arr, Σp); push!(S_arr, S)
end


μ_interest = transpose(μ_ukf_arr[131][1:2])
Σ_interest = Σ_ukf_arr[131][1:2, 1:2]

##
fid = h5open("bicycle_dataset_continuous.h5", "r")
points = read(fid["position"])[1:5000, :]
time = Float64.(read(fid["time"]))[1:5000]

##
red_mask = [t > 12.99 && t < 13.09 for t in time]
red_dots = points[red_mask, :]

##
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(1, s0, s->bicycle_dynamics(s, withNoise=false), t, drop_rate=0.0);
x = [s[1] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];
y = [s[2] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];

##
a = Axis(style="enlarge x limits=false,grid=both, no marks", axisEqualImage=true,
            xlabel="x", ylabel="y",title="Normalizing Flow Level Sets", xmin = 0, xmax = 90,
            legendPos = "north east",legendStyle="nodes = {scale = 0.75}",  view="{0}{90}")
push!(a, PGFPlots.Linear(points', style = "only marks, gray, mark options=
    {scale=0.1,fill=gray, solid, mark = *}"))
push!(a, PGFPlots.Linear(red_dots', style = "only marks,dark_red, mark options=
    {scale=0.3,fill=dark_red, solid, mark = *}"))
push!(a, plot_ellipse(μ_interest, Σ_interest, 0.68, "viridis1"))
push!(a, plot_ellipse(μ_interest, Σ_interest, 0.95, "viridis2"))
push!(a, plot_ellipse(μ_interest, Σ_interest, 0.995, "viridis3"))
#push!(a, PGFPlots.Linear(x68, y68, style = "viridis1, thick"))
#push!(a, PGFPlots.Linear(x95, y95, style = "viridis2, thick"))
#push!(a, PGFPlots.Linear(x995, y995, style = "viridis3, thick"))
push!(a, PGFPlots.Linear(x, y, style = "black, thick"))

save("figs/ukf_level_sets.pdf", a)
save("figs/ukf_level_sets.tex", a, include_preamble=true)




##
#*******************************************************************************
# NORMALIZING FLOW LEVEL SETS
#*******************************************************************************
fid = h5open("flow_level_sets.h5", "r")

x68 = Float64.(read(fid["x68"]))
y68 = Float64.(read(fid["y68"]))
x95 = Float64.(read(fid["x95"]))
y95 = Float64.(read(fid["y95"]))
x995 = Float64.(read(fid["x995"]))
y995 = Float64.(read(fid["y995"]))

close(fid)

##
fid = h5open("bicycle_dataset_continuous.h5", "r")
points = read(fid["position"])[1:5000, :]
time = Float64.(read(fid["time"]))[1:5000]

##
red_mask = [t > 12.99 && t < 13.09 for t in time]
red_dots = points[red_mask, :]

##
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(1, s0, s->bicycle_dynamics(s, withNoise=false), t, drop_rate=0.0);
x = [s[1] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];
y = [s[2] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];

##
a = Axis(style="enlarge x limits=false,grid=both, no marks", axisEqualImage=true,
            xlabel="x", ylabel="y",title="Normalizing Flow Level Sets", xmin = 0, xmax = 90,
            legendPos = "north east",legendStyle="nodes = {scale = 0.75}",  view="{0}{90}")
push!(a, PGFPlots.Linear(points', style = "only marks, gray, mark options=
    {scale=0.1,fill=gray, solid, mark = *}"))
push!(a, PGFPlots.Linear(red_dots', style = "only marks,dark_red, mark options=
    {scale=0.3,fill=dark_red, solid, mark = *}"))
push!(a, PGFPlots.Linear(x68, y68, style = "viridis1, thick"))
push!(a, PGFPlots.Linear(x95, y95, style = "viridis2, thick"))
push!(a, PGFPlots.Linear(x995, y995, style = "viridis3, thick"))
push!(a, PGFPlots.Linear(x, y, style = "black, thick"))

save("figs/normalizing_flow_level_sets.pdf", a)
save("figs/normalizing_flow_level_sets.tex", a, include_preamble=true)

##
#*******************************************************************************
# MDN LEVEL SETS
#*******************************************************************************
fid = h5open("mdn_level_sets.h5", "r")

# Note: Data is backwards?
x995 = Float64.(read(fid["x68"]))
y995 = Float64.(read(fid["y68"]))
x95 = Float64.(read(fid["x95"]))
y95 = Float64.(read(fid["y95"]))
x68 = Float64.(read(fid["x995"]))
y68 = Float64.(read(fid["y995"]))

close(fid)

##
fid = h5open("bicycle_dataset_continuous.h5", "r")
points = read(fid["position"])[1:5000, :]
time = Float64.(read(fid["time"]))[1:5000]

##
red_mask = [t > 12.99 && t < 13.09 for t in time]
red_dots = points[red_mask, :]

##
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(1, s0, s->bicycle_dynamics(s, withNoise=false), t, drop_rate=0.0);
x = [s[1] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];
y = [s[2] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];

##
a = Axis(style="enlarge x limits=false,grid=both, no marks", axisEqualImage=true,
            xlabel="x", ylabel="y",title="Normalizing Flow Level Sets", xmin = 0, xmax = 90,
            legendPos = "north east",legendStyle="nodes = {scale = 0.75}",  view="{0}{90}")
push!(a, PGFPlots.Linear(points', style = "only marks, gray, mark options=
    {scale=0.1,fill=gray, solid, mark = *}"))
push!(a, PGFPlots.Linear(red_dots', style = "only marks,dark_red, mark options=
    {scale=0.3,fill=dark_red, solid, mark = *}"))
push!(a, PGFPlots.Linear(x68, y68, style = "viridis1, thick"))
push!(a, PGFPlots.Linear(x95, y95, style = "viridis2, thick"))
push!(a, PGFPlots.Linear(x995, y995, style = "viridis3, thick"))
push!(a, PGFPlots.Linear(x, y, style = "black, thick"))

save("figs/mdn_level_sets.pdf", a)
save("figs/mdn_level_sets.tex", a, include_preamble=true)

##
#*******************************************************************************
# VAE LEVEL SETS
#*******************************************************************************
fid = h5open("vae_level_sets.h5", "r")

# Note: Data is backwards?
x995 = Float64.(read(fid["x68"]))
y995 = Float64.(read(fid["y68"]))
x95 = Float64.(read(fid["x95"]))
y95 = Float64.(read(fid["y95"]))
x68 = Float64.(read(fid["x995"]))
y68 = Float64.(read(fid["y995"]))

close(fid)

##
fid = h5open("bicycle_dataset_continuous.h5", "r")
points = read(fid["position"])[1:5000, :]
time = Float64.(read(fid["time"]))[1:5000]

##
red_mask = [t > 12.99 && t < 13.09 for t in time]
red_dots = points[red_mask, :]

##
s0 = [x0, y0, θ0, v0, ϕ0, t0]
t = lastindex(times)
τ_arr = simulate_τ(1, s0, s->bicycle_dynamics(s, withNoise=false), t, drop_rate=0.0);
x = [s[1] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];
y = [s[2] for i in 1:lastindex(τ_arr) for s in τ_arr[i]];

##
a = Axis(style="enlarge x limits=false,grid=both, no marks", axisEqualImage=true,
            xlabel="x", ylabel="y",title="Normalizing Flow Level Sets", xmin = 0, xmax = 90,
            legendPos = "north east",legendStyle="nodes = {scale = 0.75}",  view="{0}{90}")
push!(a, PGFPlots.Linear(points', style = "only marks, gray, mark options=
    {scale=0.1,fill=gray, solid, mark = *}"))
push!(a, PGFPlots.Linear(red_dots', style = "only marks,dark_red, mark options=
    {scale=0.3,fill=dark_red, solid, mark = *}"))
push!(a, PGFPlots.Linear(x68, y68, style = "viridis1, thick"))
push!(a, PGFPlots.Linear(x95, y95, style = "viridis2, thick"))
push!(a, PGFPlots.Linear(x995, y995, style = "viridis3, thick"))
push!(a, PGFPlots.Linear(x, y, style = "black, thick"))

save("figs/vae_level_sets.pdf", a)
save("figs/vae_level_sets.tex", a, include_preamble=true)

##
gru_loss = CSV.read("figs/loss/gru_loss.csv", DataFrame; header=false).Column1
rnn_loss = CSV.read("figs/loss/rnn_loss.csv", DataFrame; header=false).Column1