#approximated term kd*q by NN and forming UDE 
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
tspan=(0.0, 350.0)
datasize=50
tsteps=range(tspan[1], tspan[2], length=datasize)

## call ODEProblem function 

prob= ODEProblem(ODE, u0, tspan, p_initial)

## solve the problem
sol = solve(prob, Tsit5(), saveat=tsteps)

#analyse the result 
true_sol=Array(sol)
t=sol.t
S=Array(sol)
##################################################################################################################
#Solve the same problem using NN 

#load the packages
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
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

tspan=(0.0, 350.0)
prob_nn=ODEProblem(nn_ude!, S[:, 1], tspan, p)
##train the NN
#define predict function
function predict(θ, X= S[:,1], T=t)
    prob_nn2=remake(prob_nn, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn=Array(solve(prob_nn2, Tsit5(), saveat=t, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred=predict(θ)
    mean(abs2, S.-pred)
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
p_final=opt_sol3.u
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
nn_ude_pred_sol=predict(p_final, S[:,1], t)

#plot the results
#plot UDE approximated solution
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

plot_true_UDE_approximation=plot(t, true_sol[1, :],  seriestype=:line, linestyle=:dash, lw=5, color=:blue, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(ts, transpose(nn_ude_pred_sol), seriestype=:line, lw=3, label="UDE predicted data", color=:red)
savefig("true data_predicted UDE.png")
savefig("true data_predicted UDE.svg")
# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_sol[1, :])'
Ȳ=true_missing_terms
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
# Plot the true missing terms and the UDE approximations
plot(t, true_missing_terms',seriestype=:scatter, marker=:circle, markersize=7.0, color=:magenta, alpha=0.8,  label="Actual missing term", xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="Missing Desorption Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot_reconstruction = plot!(t, nn_pred_missing_terms', seriestype=:line, lw=5, color=:cyan, label="UDE Predicted missing term")
savefig("true missing term_UDE prediction.png")
savefig("true missing term_UDE prediction.svg")
#plot the error
plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error Plot",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black,  label = nothing, color = :seagreen, lw=3,  legendfontsize=14)
savefig("error.png")
savefig("error.svg")
#all in one
using Plots.PlotMeasures
plot_size = (1200, 800)
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

plot_overall = plot(size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, plot_true_UDE_approximation, plot_reconstruction, plot_error)
savefig("true_UDE_missing_terms_error.png")
savefig("true_UDE_missing_terms_error.svg")
savefig("true_UDE_missing_terms_error.pdf")
