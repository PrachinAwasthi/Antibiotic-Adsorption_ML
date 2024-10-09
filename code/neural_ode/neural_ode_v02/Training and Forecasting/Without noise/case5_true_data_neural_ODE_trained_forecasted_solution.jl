#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~case5- 10%Training data & 90% Testing data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
p0=[0.01, 0.001, 5.0, 100.0]
tspan=(0.0, 35.0)
datasize=5
tsteps=range(tspan[1], tspan[2], length=datasize)

## call ODEProblem function 
prob= ODEProblem(ODE, u0, tspan, p0)

## solve the problem
sol = solve(prob, Tsit5(), saveat=tsteps)

#analyse the result 
true_sol=Array(sol)
t=sol.t
S=Array(sol)

#~~~~~~~~~~~~~~~~~~~~~~~~~~Neural ODE solution~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load the packages
using OrdinaryDiffEq, ComponentArrays
using Lux, Flux, DiffEqFlux, Random

##Define random no and seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng,0)

## define neural Network
nn= Lux.Chain(Lux.Dense(1,16,sigmoid), Lux.Dense(16,16,sigmoid), Lux.Dense(16,1))

## define parameters and variables of NN
p,st=Lux.setup(rng,nn)

nnode_sol=NeuralODE(nn, tspan, Tsit5(), saveat=tsteps)
## define a prediction function which uses neuralODE() with initial conditions and parameters


function nnode_predict(p)
    Array(nnode_sol(u0, p, st)[1])
end


## Define a loss function
function nnode_loss(p)
    pred=nnode_predict(p)
    loss=sum(abs2, true_sol.-pred)
    return loss, pred
end

## define the callback function 
losses_original= Float32[]
callback= function (p, l, pred; doplot=true)
    push!(losses_original, l)
    if length(losses_original)%50==0
    println("Current loss after $(length(losses_original)) iterations: $(losses_original[end])")
    end
    return false

if  length(losses_original) % 50 == 0
        plt = scatter(t, true_sol[1, :], label="actual data")
        scatter!(plt, t, pred[1, :], label="prediction")
        display(plot(plt))
    end
    return false
end
pinit = ComponentArray(p)

## optimize the loss function
using Optimization, OptimizationOptimisers, OptimizationOptimJL

## define a automatic differentiation
adtype=Optimization.AutoZygote()

## create the optimization function and problem
opt_fn= Optimization.OptimizationFunction((x,p)->nnode_loss(x), adtype)
opt_prob= Optimization.OptimizationProblem(opt_fn, pinit)

## solve the optimization problem using optimiser
opt_sol= Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.001), maxiters=1000, callback=callback, allow_f_increases=false)

## Again optimizing to converge the predicted solution
opt_prob2=remake(opt_prob, u0 = opt_sol.u)
opt_sol2= Optimization.solve(opt_prob2, OptimizationOptimisers.Adam(0.001), maxiters=10000, callback=callback, allow_f_increases=false)

#remake the optimization problem
opt_prob3=remake(opt_prob2, u0 = opt_sol2.u)
#solve the problem
opt_sol3=Optimization.solve(opt_prob3, LBFGS(), callback=callback, maxiters=2000)

p_trained=opt_sol3.u
nnode_pred_sol= nnode_predict(opt_sol3.u)

###PLot  the results
#using Plots.PlotMeasures
#plot_size = (1200, 600)
#left_margin = 20px
#right_margin = 20px
#top_margin = 20px
#bottom_margin = 20px

#plot(t, true_sol[1, :],  seriestype=:scatter, marker=:circle,markersize=7.0, lw=5, color=:blue, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)",  labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(tsteps, nnode_pred_sol[1, :], seriestype=:line, lw=5, label="Neural ODE predicted data", color=:red)
#savefig("true data_neural ODE predicted data_case5.png")

###############################################Testing####################################################
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
u0_testing=[true_sol[end]]
p_initial=[0.01, 0.001, 5.0, 100.0]
testing_tspan = (35, 350.0)
testing_datasize = 45
testing_tsteps = range(testing_tspan[1], testing_tspan[2], length=testing_datasize)

## call ODEProblem function 
prob_testing= ODEProblem(ODE, u0_testing, testing_tspan, p_initial)

## solve the problem
sol_testing= solve(prob_testing, Tsit5(), saveat=testing_tsteps)

#analyse the result 
true_sol_testing=Array(sol_testing)
t_testing=sol_testing.t
S_testing=Array(sol_testing)
S=Array(sol)
#combined_true = vcat(S[1, :], S_testing[1, 2:end])

#combined_tsteps = vcat(t, testing_tsteps[2:end])

#plot(tsteps, true_sol[1, :], tickfontsize=15, seriestype=:scatter, marker=:circle, markersize=7.0, lw=3, alpha=0.5, color=:red, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(testing_tsteps,  true_sol_testing[1, :] , seriestype=:scatter, marker=:diamond, markersize=7.0, lw=3, color=:red, label="True forecast data")

#savefig("true data_trained_tested_case5.png")
#~~~~~~~~~~~~~~~~~~~~~~~~~~Neural ODE solution_Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load the packages
using OrdinaryDiffEq, ComponentArrays
using Lux, Flux, DiffEqFlux, Random

##Define random no and seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng,0)

testing_tspan = (35, 350.0)
testing_datasize = 45
testing_tsteps = range(testing_tspan[1], testing_tspan[2], length=testing_datasize)
u0= nnode_pred_sol[:, end]

## define neural Network
nn= Lux.Chain(Lux.Dense(1,16,sigmoid), Lux.Dense(16,16,sigmoid), Lux.Dense(16,1))

## define parameters and variables of NN
p,st=Lux.setup(rng,nn)

nnode_sol=NeuralODE(nn, testing_tspan, Tsit5(), saveat=testing_tsteps)
## define a prediction function which uses neuralODE() with initial conditions and parameters

u0= nnode_pred_sol[:, end]
p_trained=opt_sol3.u
p=p_trained
function nnode_extended_predict(p)
    Array(nnode_sol(u0, p, st)[1])
end


## Define a loss function
function nnode_loss(p)
    pred_testing=nnode_extended_predict(p)
    loss=sum(abs2, true_sol_testing.-pred_testing)
    return loss, pred_testing
end

## define the callback function 
#losses_original= Float32[]
losses_testing=Float64[]
callback= function (p, l, pred; doplot=true)
    push!(losses_testing, l)
    if length(losses_testing)%50==0
    println("Current loss after $(length(losses_testing)) iterations: $(losses_testing[end])")
    end
    return false
end

pinit = ComponentArray(p)

## optimize the loss function
using Optimization, OptimizationOptimisers, OptimizationOptimJL

## define a automatic differentiation
adtype=Optimization.AutoZygote()

## create the optimization function and problem
opt_fn_testing= Optimization.OptimizationFunction((x,p)->nnode_loss(x), adtype)
opt_prob_testing= Optimization.OptimizationProblem(opt_fn_testing, pinit)

## solve the optimization problem using optimiser
opt_sol_testing= Optimization.solve(opt_prob_testing, OptimizationOptimisers.Adam(0.001), maxiters=1000, callback=callback, allow_f_increases=false)

## Again optimizing to converge the predicted solution
opt_prob2_testing=remake(opt_prob_testing, u0 = opt_sol_testing.u)
opt_sol2_testing= Optimization.solve(opt_prob2_testing, OptimizationOptimisers.Adam(0.001), maxiters=10000, callback=callback, allow_f_increases=false)

#remake the optimization problem
opt_prob3_testing=remake(opt_prob2_testing, u0 = opt_sol2_testing.u)
#solve the problem
opt_sol3_testing=Optimization.solve(opt_prob3_testing, LBFGS(), callback=callback, maxiters=2000)

p_tested=opt_sol3_testing.u
nnode_pred_sol_testing= nnode_predict(opt_sol3_testing.u)

###PLot  the results
#using Plots.PlotMeasures
#lot_size = (1200, 600)
#left_margin = 20px
#right_margin = 20px
#top_margin = 20px
#bottom_margin = 20px

#plot(tsteps, nnode_pred_sol[1, :], tickfontsize=12, seriestype=:line,  lw=3, label="Neural ODE predicted data", alpha=0.5, color=:blue, xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(testing_tsteps, nnode_pred_sol_testing[1, :],seriestype=:line,  linestyle=:dash,  lw=3,  color=:blue, label="Neural ODE forecast predicted data")
#savefig("Neural ODE_trained_tested_case5.png")

##########################all in one##############

#plot(tsteps, true_sol[1, :], tickfontsize=15, seriestype=:scatter, marker=:circle, markersize=7.0, lw=3, alpha=0.5, color=:red, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(tsteps, nnode_pred_sol[1, :], tickfontsize=12, seriestype=:line,  lw=3, label="Neural ODE predicted data", alpha=0.5 , color=:blue, xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(testing_tsteps,  true_sol_testing[1, :] , seriestype=:scatter, marker=:diamond, markersize=7.0, lw=3, color=:red, label="True forecast data")
#plot!(testing_tsteps, nnode_pred_sol_testing[1, :],seriestype=:line,  linestyle=:dash,  lw=3,  color=:blue, label="Neural ODE forecast predicted data")
#savefig("true data_neural ODE_trained_tested_case5.png")

using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px


plot(tsteps, true_sol[1, :], seriestype=:scatter, marker=:circle, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)",  labelcolor=:darkblack, title="Langmuir Adsorption", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(testing_tsteps,  true_sol_testing[1, :] , seriestype=:scatter, marker=:circle,markersize=10.0, alpha=0.5, color=:red, label="Testing data")
plot!(tsteps, nnode_pred_sol',seriestype=:line, lw=3, label="Predicted data", color=:blue)
plot!(testing_tsteps, nnode_pred_sol_testing', seriestype=:line, lw=3, label="Forecasted data", color=:red)
savefig("2_true data_neural ODE_trained_tested_case5.png")