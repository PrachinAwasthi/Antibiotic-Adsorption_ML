#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~case4- 30%Training data & 70% Testing data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################Training####################################################
##load the packages
using DifferentialEquations, Plots
using LinearAlgebra, Statistics, StableRNGs
##define variables and parameters
#α=ka, β=kb, γ=C, δ=Qmax
α=(0.01)
β=(0.001)
γ=5.0
δ=100.0
p=[α, β, γ, δ]
#define equation in terms of function

function ODE(du,u,p,t)
   du[1]= α*γ*(δ-u[1])-β*u[1]
end

#define initial values
u0=[0.0]
p_initial=[0.01, 0.001, 5.0, 100.0]

tspan=(0.0, 105.0)
datasize=15
tsteps=range(tspan[1], tspan[2], length=datasize)

## call ODEProblem function 
prob= ODEProblem(ODE, u0, tspan, p_initial)

## solve the problem
sol = solve(prob, Tsit5(), saveat=tsteps)

#analyse the result 
true_sol=Array(sol)
t=sol.t
S=Array(sol)

#add high noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-2
xₙ= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_noisy=Array(xₙ)
S_noisy=Array(xₙ)
##################################################################################################################
#Solve the same problem using NN 

#load the packages
using OrdinaryDiffEq, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Flux, DiffEqFlux, JLD, Random, Zygote


##Define random no and seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng,0)

## define neural Network
activation= sigmoid
nn= Lux.Chain(Lux.Dense(1,16,activation), Lux.Dense(16,16,activation), Lux.Dense(16,1))
p,st=Lux.setup(rng,nn)

#define equation using NN
function nn_ude(du, u, p, t, p_initial)
    u_pred = nn(u, p, st)[1]
    du[1]= p_initial[1]*p_initial[3]*(p_initial[4]-u[1])-u_pred[1]
end

#define NN in ODEProblem
nn_ude!(du, u, p,t)=nn_ude(du, u, p, t, p_initial)

tspan=(0.0, 105.0)
prob_nn=ODEProblem(nn_ude!, S_noisy[:, 1], tspan, p)
##train the NN
#define predict function
function predict(θ, X= S_noisy[:,1], T=t)
    prob_nn2=remake(prob_nn, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn=Array(solve(prob_nn2, Tsit5(), saveat=t, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred=predict(θ)
    mean(abs2, xₙ.-pred)
end

#define callback function
losses_original=Float64[]
callback = function(p,l)
    push!(losses_original, l)
    if length(losses_original) % 50 == 0
        println("Current loss after $(length(losses_original)) iterations: $(losses_original[end])")
    end
    return false
end

#define optimization 
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

#define optimization problem
opt_prob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
opt_sol= Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.001), maxiters=1000, callback=callback, allow_f_increases=false)

## Again optimizing to converge the predicted solution
opt_prob2=remake(opt_prob, u0 = opt_sol.u)
opt_sol2= Optimization.solve(opt_prob2, OptimizationOptimisers.Adam(0.001), maxiters=10000, callback=callback, allow_f_increases=false)

#remake the optimization problem and solve the problem
opt_prob3=remake(opt_prob2, u0 = opt_sol2.u)
opt_sol3=Optimization.solve(opt_prob3, LBFGS(), callback=callback, maxiters=2000)

#analyse the results
p_trained=opt_sol3.u
t=sol.t
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
nn_ude_pred_sol_noisy=predict(p_trained, S_noisy[:,1], t)

#plot the results
#plot UDE approximated solution
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

plot_true_UDE_approximation=plot(t, true_sol_noisy[1, :],  seriestype=:scatter, marker=:circle, markersize=7.0, lw=5, alpha=0.5, color=:red, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)",labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(t, transpose(nn_ude_pred_sol_noisy), seriestype=:line, lw=5, label="UDE predicted data", alpha=0.5, color=:blue)
savefig("high noise data traied_case4.png")
savefig("high noise data traied_case4.svg")

###############################################Testing####################################################
##testing for true solution
α=(0.01)
β=(0.001)
γ=5.0
δ=100.0
p=[α, β, γ, δ]
#define equation in terms of function

function ODE(du,u,p,t)
   du[1]= α*γ*(δ-u[1])-β*u[1]
end

#define initial values
u0_testing=[true_sol_noisy[end]]
p_initial=[0.01, 0.001, 5.0, 100.0]
testing_tspan = (105, 350.0)
testing_datasize = 35
testing_tsteps = range(testing_tspan[1], testing_tspan[2], length=testing_datasize)

## call ODEProblem function 
prob_testing= ODEProblem(ODE, u0_testing, testing_tspan, p_initial)

## solve the problem
sol_testing = solve(prob_testing, Tsit5(), saveat=testing_tsteps)

#analyse the result 
true_sol_testing=Array(sol_testing)
t_testing=sol_testing.t
S_testing=Array(sol_testing)

#add high noise interms of mean
x̄= mean(true_sol_testing)
noise_magnitude = 5e-2
xₙ_testing= true_sol_testing.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol_testing), size(true_sol_testing))
true_sol_testing_noisy=Array(xₙ_testing)
S_testing_noisy=Array(xₙ_testing)

combined_true = vcat(S_noisy[1, :], S_testing_noisy[1, 2:end])

combined_tsteps = vcat(t, testing_tsteps[2:end])

plot(tsteps, true_sol_noisy[1, :], tickfontsize=15, seriestype=:scatter, marker=:circle, markersize=7.0, lw=3, alpha=0.5, color=:red, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(testing_tsteps,  true_sol_testing_noisy[1, :] , seriestype=:scatter, marker=:diamond, markersize=7.0, lw=3, color=:red, label="True forecast data")

savefig("high noise data_trained_tested_case4.png")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
####################NNUDE prediction for testing#############################################################
# Extend the timespan for prediction
testing_tspan = (105, 350.0)
testing_datasize = 35
testing_tsteps = range(testing_tspan[1], testing_tspan[2], length=testing_datasize)

##Define random no and seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng,0)

## define neural Network
activation= sigmoid
nn= Lux.Chain(Lux.Dense(1,16,activation), Lux.Dense(16,16,activation), Lux.Dense(16,1))
p,st=Lux.setup(rng,nn)

#define equation using NN
function nn_ude(du, u, p, t, p_initial)
    u_pred = nn(u, p, st)[1]
    du[1]= p_initial[1]*p_initial[3]*(p_initial[4]-u[1])-u_pred[1]
end

#define NN in ODEProblem
nn_ude!(du, u, p,t)=nn_ude(du, u, p, t, p_initial)

testing_tspan = (105, 350.0)
S_testing_noisy=Array(xₙ_testing)
p_trained=opt_sol3.u
prob_nn_testing=ODEProblem(nn_ude!, S_testing_noisy[:, 1], testing_tspan, p_trained)

##train the NN
#define predict function
function extended_predict(θ, X=S_testing_noisy[:, 1], T=testing_tsteps)
    prob_nn2_testing=remake(prob_nn_testing, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn_testing=Array(solve(prob_nn2_testing, Tsit5(), saveat=T, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred_testing=extended_predict(θ)
    mean(abs2, S_testing_noisy.-pred_testing)
end

#define callback function
losses_testing=Float64[]
callback = function(p,l)
    push!(losses_testing, l)
    if length(losses_testing) % 50 == 0
        println("Current loss after $(length(losses_testing)) iterations: $(losses_testing[end])")
    end
    return false
end

#define optimization 
adtype = Optimization.AutoZygote()
optf_testing = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

#define optimization problem
opt_prob_testing = Optimization.OptimizationProblem(optf_testing, ComponentVector{Float64}(p))
opt_sol_testing= Optimization.solve(opt_prob_testing, OptimizationOptimisers.Adam(0.001), maxiters=1000, callback=callback, allow_f_increases=false)

## Again optimizing to converge the predicted solution
opt_prob2_testing=remake(opt_prob_testing, u0 = opt_sol_testing.u)
opt_sol2_testing= Optimization.solve(opt_prob2_testing, OptimizationOptimisers.Adam(0.001), maxiters=10000, callback=callback, allow_f_increases=false)

#remake the optimization problem and solve the problem
opt_prob3_testing=remake(opt_prob2_testing, u0 = opt_sol2_testing.u)
opt_sol3_testing=Optimization.solve(opt_prob3_testing, LBFGS(), callback=callback, maxiters=2000)

#analyse the results
p_tested=opt_sol3_testing.u

nn_ude_pred_sol_testing_noisy=extended_predict(p_tested, S_testing_noisy[:, 1], testing_tsteps)

combined_pred = vcat(nn_ude_pred_sol_noisy[1, :], nn_ude_pred_sol_testing_noisy[1, 2:end])

combined_tsteps = vcat(t, testing_tsteps[2:end])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

plot(tsteps, nn_ude_pred_sol_noisy', tickfontsize=12, seriestype=:line,  lw=3, label="UDE predicted data", alpha=0.5, color=:blue, xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(testing_tsteps, nn_ude_pred_sol_testing_noisy', seriestype=:line,  linestyle=:dash,  lw=3,  color=:blue, label="UDE forecast predicted data")
savefig("UDE_trained_tested_case4.png")

##########################all in one##############

plot(tsteps, true_sol_noisy[1, :], tickfontsize=15, seriestype=:scatter, marker=:circle, markersize=7.0, lw=3, alpha=0.5, color=:red, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(tsteps, nn_ude_pred_sol_noisy', tickfontsize=12, seriestype=:line,  lw=5, label="UDE predicted data", alpha=0.5, color=:blue)
plot!(testing_tsteps,  true_sol_testing_noisy[1, :] , seriestype=:scatter, marker=:diamond, markersize=7.0, lw=3, color=:red, label="True forecast data")
plot!(testing_tsteps, nn_ude_pred_sol_testing_noisy', seriestype=:line, lw=5,  color=:blue, label="UDE forecast predicted data")
savefig("high noise data_UDE_trained_tested_case4.png")