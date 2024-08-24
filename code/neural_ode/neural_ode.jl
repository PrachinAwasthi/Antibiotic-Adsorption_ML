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
tspan=(0.0, 350.0)
datasize=50
tsteps=range(tspan[1], tspan[2], length=datasize)

## call ODEProblem function 
prob= ODEProblem(ODE, u0, tspan, p0)

## solve the problem
sol = solve(prob, Tsit5(), saveat=tsteps)

#analyse the result 
true_sol=Array(sol)
t=sol.t

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

nnode_pred_sol= nnode_predict(opt_sol3.u)

###PLot  the results
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

plot(t, true_sol[1, :], seriestype=:line, lw=5, color=:red, alpha=0.5, label="True Solution", xlabel="Time (min)", ylabel="Adsorption capacity (mg/g)", title="Langmuir Adsorption Model", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright)
plot!(tsteps, nnode_pred_sol[1, :], seriestype=:scatter, marker=:circle, markersize=5.0, lw=3, label="Neural prediction", color=:blue)
savefig("plot.png")