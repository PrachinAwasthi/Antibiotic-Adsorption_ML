#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`Low   noise  solution-  5e-3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#add low noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-3
xₙ= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_noisy=Array(xₙ)

#Solve the same problem using NN 

#load the packages
using OrdinaryDiffEq,  DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
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
prob_nn=ODEProblem(nn_ude!, xₙ[:, 1], tspan, p)
##train the NN
#define predict function
function predict(θ, X= xₙ[:,1], T=t)
    prob_nn2=remake(prob_nn, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn=Array(solve(prob_nn2, Tsit5(), saveat=t, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred=predict(θ)
    mean(abs2, xₙ.-pred)
end

#define callback function
losses_noisy=Float64[]
callback = function(p,l)
    push!(losses_noisy, l)
    if length(losses_noisy) % 50 == 0
        println("Current loss after $(length(losses_noisy)) iterations: $(losses_noisy[end])")
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
nn_ude_pred_noisy_sol=predict(p_final, xₙ[:,1], t)

#plot the results
#plot UDE approximated solution

using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

#plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :],   seriestype=:line, linestyle=:dash, lw=5, color=:brown, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(ts, transpose(nn_ude_pred_noisy_sol),  seriestype=:line, lw=3, label="UDE predicted data", color=:lightseagreen)
#savefig("low noise_true noisy data_predicted UDE.png")
#savefig("low noise_true noisy data_predicted UDE.svg")
plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :],  seriestype=:scatter, marker=:circle,markersize=7.0, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(ts, transpose(nn_ude_pred_noisy_sol),  seriestype=:line, lw=3, label="Predicted data", color=:black)
savefig("2_low noise_true noisy data_predicted UDE.png")


# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_noisy_sol[1, :])'
Ȳ=true_missing_terms
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
# Plot the true missing terms and the UDE approximations
#plot(t, true_missing_terms',seriestype=:scatter, marker=:circle, markersize=7.0, color=:magenta, alpha=0.8,  label="Actual missing term", xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="Missing Desorption Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:topright, grid=false, legendfontsize=14)
#plot_reconstruction = plot!(t, nn_pred_missing_terms', seriestype=:line, lw=5, color=:cyan, label="UDE Predicted missing term")
#savefig("true noisy missing term_UDE prediction.png")
#savefig("true noisy missing term_UDE prediction.svg")

plot(t, true_missing_terms',marker=:circle,seriestype=:scatter,markersize=7.0, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="UDE Missing Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:topright, grid=false, legendfontsize=14)
plot_reconstruction = plot!(t, nn_pred_missing_terms',seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")

savefig("2_true noisy missing term_UDE prediction.png")

#plot the error
#plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error Plot",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black, label = nothing, color = :seagreen, lw=3,  legendfontsize=14)

#savefig("error.png")
#savefig("error.svg")

plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black,  label = nothing, color = :orange, lw=3,  legendfontsize=14)
savefig("2_error.png")

#all in one
using Plots.PlotMeasures
plot_size = (1200, 800)
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

#plot_overall = plot(size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, plot_true_noisy_UDE_approximation, plot_reconstruction, plot_error)
#savefig("true noisy_UDE_missing_terms_error.png")
#savefig("true noisy_UDE_missing_terms_error.svg")
#savefig("true noisy_UDE_missing_terms_error.pdf")
plot_overall = plot(size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, plot_true_noisy_UDE_approximation, plot_reconstruction, plot_error)
savefig("2_true noisy_UDE_missing_terms_error.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`High   noise  solution-  5e-2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#add low noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-2
xₙ= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_noisy=Array(xₙ)

#Solve the same problem using NN 

#load the packages
using OrdinaryDiffEq,  DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
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
prob_nn=ODEProblem(nn_ude!, xₙ[:, 1], tspan, p)
##train the NN
#define predict function
function predict(θ, X= xₙ[:,1], T=t)
    prob_nn2=remake(prob_nn, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn=Array(solve(prob_nn2, Tsit5(), saveat=t, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred=predict(θ)
    mean(abs2, xₙ.-pred)
end

#define callback function
losses_noisy=Float64[]
callback = function(p,l)
    push!(losses_noisy, l)
    if length(losses_noisy) % 50 == 0
        println("Current loss after $(length(losses_noisy)) iterations: $(losses_noisy[end])")
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
nn_ude_pred_high_noisy_sol=predict(p_final, xₙ[:,1], t)

#plot the results
#plot UDE approximated solution

using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

#plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :],   seriestype=:line, linestyle=:dash, lw=5, color=:brown, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(ts, transpose(nn_ude_pred_high_noisy_sol),  seriestype=:line, lw=3, label="UDE predicted data", color=:lightseagreen)
#savefig("high noise_true noisy data_predicted UDE.png")
#savefig("high noise_true noisy data_predicted UDE.svg")
plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :], seriestype=:scatter, marker=:circle,markersize=7.0, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(ts, transpose(nn_ude_pred_high_noisy_sol),seriestype=:line, lw=3, label="Predicted data", color=:black)
savefig("2_high noise_true noisy data_predicted UDE.png")




# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_high_noisy_sol[1, :])'
Ȳ=true_missing_terms
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_high_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
# Plot the true missing terms and the UDE approximations
#plot(t, true_missing_terms',seriestype=:scatter, marker=:circle, markersize=7.0, color=:magenta, alpha=0.8,  label="Actual missing term", xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="Missing Desorption Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot_reconstruction = plot!(t, nn_pred_missing_terms', seriestype=:line, lw=5, color=:cyan, label="UDE Predicted missing term")
#savefig("high noise_true noisy missing term_UDE prediction.png")
#savefig("high noise_true noisy missing term_UDE prediction.svg")

plot(t, true_missing_terms',seriestype=:scatter, marker=:circle,markersize=7.0, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="UDE Missing Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot_reconstruction = plot!(t, nn_pred_missing_terms',  seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")

savefig("2_high noise_true noisy missing term_UDE prediction.png")
#plot the error
#plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error Plot",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black, label = nothing, color = :seagreen, lw=3,  legendfontsize=14)
#savefig("high error.png")
#savefig("high error.svg")

plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black,  label = nothing, color = :orange, lw=3,  legendfontsize=14)
savefig("2_high error.png")
#all in one
using Plots.PlotMeasures
plot_size = (1200, 800)
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

plot_overall = plot(size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, plot_true_noisy_UDE_approximation, plot_reconstruction, plot_error)
savefig("2_high noise_true noisy_UDE_missing_terms_error.png")
#savefig("high noise_true noisy_UDE_missing_terms_error.svg")
#savefig("high noise_true noisy_UDE_missing_terms_error.pdf")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`Very High   noise  solution-  5e-1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#add low noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-1
xₙ= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_noisy=Array(xₙ)

#Solve the same problem using NN 

#load the packages
using OrdinaryDiffEq,  DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
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
prob_nn=ODEProblem(nn_ude!, xₙ[:, 1], tspan, p)
##train the NN
#define predict function
function predict(θ, X= xₙ[:,1], T=t)
    prob_nn2=remake(prob_nn, u0=X, p=θ, tspan=(T[1], T[end]))
    sol_nn=Array(solve(prob_nn2, Tsit5(), saveat=t, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

#define loss(MSE) function 
function loss(θ)
    pred=predict(θ)
    mean(abs2, xₙ.-pred)
end

#define callback function
losses_noisy=Float64[]
callback = function(p,l)
    push!(losses_noisy, l)
    if length(losses_noisy) % 50 == 0
        println("Current loss after $(length(losses_noisy)) iterations: $(losses_noisy[end])")
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
nn_ude_pred_very_high_noisy_sol=predict(p_final, xₙ[:,1], t)

#plot the results
#plot UDE approximated solution

using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

#plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :],   seriestype=:line, linestyle=:dash, lw=5, color=:brown, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot!(ts, transpose(nn_ude_pred_very_high_noisy_sol),  seriestype=:line, lw=3, label="UDE predicted data", color=:lightseagreen)
#savefig("very high noise_true noisy data_predicted UDE.png")
#savefig("very high noise_true noisy data_predicted UDE.svg")
plot_true_noisy_UDE_approximation=plot(t, true_sol_noisy[1, :], seriestype=:scatter, marker=:circle,markersize=7.0, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(ts, transpose(nn_ude_pred_very_high_noisy_sol),  seriestype=:line, lw=3, label="Predicted data", color=:black)
savefig("2_very high noise_true noisy data_predicted UDE.png")

# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_very_high_noisy_sol[1, :])'
Ȳ=true_missing_terms
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_very_high_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
# Plot the true missing terms and the UDE approximations
#plot(t, true_missing_terms',seriestype=:scatter, marker=:circle, markersize=7.0, color=:magenta, alpha=0.8,  label="Actual missing term", xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="Missing Desorption Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
#plot_reconstruction = plot!(t, nn_pred_missing_terms', seriestype=:line, lw=5, color=:cyan, label="UDE Predicted missing term")

#savefig("very high noise_true noisy missing term_UDE prediction.png")
#savefig("very high noise_true noisy missing term_UDE prediction.svg")

plot(t, true_missing_terms',seriestype=:scatter,marker=:circle,markersize=7.0, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)", labelfontsize=14, labelcolor=:black, title="UDE Missing Term",titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot_reconstruction = plot!(t, nn_pred_missing_terms',  seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")
savefig("2_very high noise_true noisy missing term_UDE prediction.png")
#plot the error
#plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error Plot",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black, label = nothing, color = :seagreen, lw=3,  legendfontsize=14)
#savefig("very high error.png")
#savefig("very high error.svg")

plot_error=plot(t, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, title="Error",  xlabel = "Time (min)", ylabel = "Error", labelfontsize=12, labelcolor=:black,  label = nothing, color = :orange, lw=3,  legendfontsize=14)
savefig("2_very high error.png")
#all in one
using Plots.PlotMeasures
plot_size = (1200, 800)
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

plot_overall = plot(size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, plot_true_noisy_UDE_approximation, plot_reconstruction, plot_error)
#savefig("very high noise_true noisy_UDE_missing_terms_error.png")
#savefig("very high noise_true noisy_UDE_missing_terms_error.svg")
#savefig("very high noise_true noisy_UDE_missing_terms_error.pdf")
savefig("2_very high noise_true noisy_UDE_missing_terms_error.png")