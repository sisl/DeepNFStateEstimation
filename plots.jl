#=
plots.jl:
    A file containing functions for reading stored experimental data and
    generating plots of level sets
=#

##
#*******************************************************************************
# PACKAGES AND SETUP
#*******************************************************************************
using Distributions
using HDF5
using LinearAlgebra
using PGFPlots

# Color-blind color palette
define_color("pastelRed",       [245,97,92])
define_color("pastelBlue",      [0,114,178])
define_color("pastelGreen",     [0,158,115])
define_color("pastelPurple",    [135,112,254])

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
fid = h5open("normalizing_flow_plot.h5", "r")
points = read(fid["position"])
time = Float64.(read(fid["time"]))
x = read(fid["x"])[1,:]
y = read(fid["y"])[:,1]
z = transpose(read(fid["z"]))
close(fid)

red_mask = [t > 12.8 && t < 13.1 for t in time]
red_dots = [points[:,i] for i in 1:length(red_mask) if red_mask[i] == 1]
red_x = [p[1] for p in red_dots]
red_y = [p[2] for p in red_dots]

##
a = Axis(style="enlarge x limits=false,grid=both, no marks", axisEqual=true,
            xlabel="x", ylabel="y",title="Bicycle Model Rollouts",
            legendPos = "north east",legendStyle="nodes = {scale = 0.75}",  view="{0}{90}")
p1 = PGFPlots.Linear(points, style = "only marks, gray, mark options=
    {scale=0.25,fill=gray, solid, mark = *}")
p2 = PGFPlots.Linear(red_x, red_y, style = "only marks,red, mark options=
    {scale=0.35,fill=red, solid, mark = *}")
p3 = Plots.Contour(z, x, y, contour_style ="draw color = blue, handler/.style=smooth,labels = false")


push!(a, p1); push!(a, p2); push!(a, p3);
push!(a, plot_ellipse(μ_interest, Σ_interest, 0.85, "pastelPurple"))
save("figs/normalizing_flow_result.pdf", a)
save("figs/normalizing_flow_result.tex", a, include_preamble=false)
