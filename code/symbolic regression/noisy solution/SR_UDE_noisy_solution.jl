#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~low noise~~~~~~~~~~~~~~~~~~~~~~~~~~#
#load the packages
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics, StableRNGs
using ComponentArrays, Lux, Flux, DiffEqFlux, Random, Zygote
using Symbolics, DifferentialEquations, SciMLSensitivity, Plots

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
########solve with nn as UDE#####
#Solve the same problem using NN 
#approximated term kd*q by NN and forming UDE 
##load the packages

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
function predict(θ, X= X= xₙ[:,1], T=t)
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
# Define the output file path
output_file = "UDE_low noise_loss.txt"
# Open the file once at the start of the code
file = open(output_file, "w")
# Write the header to the file
println(file, "Iteration, low noise Loss")
callback = function(p,l)
    push!(losses_noisy, l)
    # Write the current iteration and loss to the file
    println(file, "$(length(losses_noisy)), $l")
    # Flush the file buffer to make sure the data is saved
    flush(file)
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
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
p_final=opt_sol3.u
nn_ude_pred_noisy_sol=predict(p_final, xₙ[:,1], t)
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms

####################################Symbolic regression####################################
## Start the automatic discovery
@variables t x(t) 
u = [x]
basis = Basis(polynomial_basis(u, 5), u, iv = t)

problem = DirectDataDrivenProblem(nn_ude_pred_noisy_sol, nn_pred_missing_terms)
println(problem)
@show size(problem)
# Create the thresholds which should be used in the search process
λ = exp10.(-3:0.01:5)
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
dddsol2 = solve(problem, basis, opt, maxiter = 10000, progress = true)
println(get_basis(dddsol2))
dddsol2_basis=get_basis(dddsol2)
get_parameter_values(dddsol2_basis)


p=[α, β, γ, δ]
function recovered_ude(du, u, p, t)
    u_rec = dddsol2_basis(u, p)
    du[1]= p_initial[1]*p_initial[3]*(p_initial[4]-u[1])-u_rec[1]
end

estimation_prob = ODEProblem(recovered_ude, u0, tspan, get_parameter_values(dddsol2_basis))
estimate = solve(estimation_prob, Tsit5(), saveat = tsteps)
u_rec = dddsol2_basis(u, get_parameter_values(dddsol2_basis))
# Define the output file path
sr_output_file = "SR_UDE_low_noise_initial.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
    println(file, "Equation: ", dddsol2_basis)
    println(file, "Parameters: ", get_parameter_values(dddsol2_basis))
    println(file, "Equation with parameters: ", u_rec)
end
################################optimizing the loss###############################
function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_noisy_sol)))
    sum(abs2, nn_pred_missing_terms .- Y)
end

optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(dddsol2_basis))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 10000)

# Look at long term prediction
tspan=(0.0, 350.0)
datasize=50
tsteps=range(tspan[1], tspan[2], length=datasize)
estimation_prob = ODEProblem(recovered_ude, u0, tspan, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = tsteps) # Using higher tolerances here results in exit of julia
u_rec_updated = dddsol2_basis(u, parameter_res)

# Define the output file path
sr_output_file = "SR_UDE_low_noise_final.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
# Write the header to the file
println(file, "Equation: ", dddsol2_basis)
println(file, "Updated Parameters: ", parameter_res)
println(file, "Equation with updated parameters: ", u_rec_updated)
end
#####################plotting###########################
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px

ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
t= sol.t
plot_true_UDE_approximation=plot(ts, true_sol_noisy[1, :], seriestype=:scatter, marker=:circle, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelcolor=:darkblack, title="Langmuir Adsorption",  size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(ts, transpose(nn_ude_pred_noisy_sol), seriestype=:line, lw=3, label="Predicted data", color=:black)
plot!(ts, estimate_long[1,:], seriestype=:line, linestyle=:dash,   lw=4, label="Regression data", color=:lightgreen)
savefig("01_low noise_true data_UDE_reg_pred_data.png")


######PLoting of missing terms#######
# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_noisy_sol[1, :])'
Ȳ=true_missing_terms
println(Ȳ)
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
println(Ŷ)
# compute regression missing terms
reg_pred_missing_terms = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_sol)))
Y=reg_pred_missing_terms
println(Y)

# Plot
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px
plot(t, true_missing_terms', seriestype=:scatter, marker=:circle, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)",  labelcolor=:black, title="UDE Missing Term", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(t, nn_pred_missing_terms', seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")
plot!(t, reg_pred_missing_terms', seriestype=:line, lw=3, color=:blue, label="SR Approximation")
savefig("01_low noise_true_UDE_reg_pred_missing_terms.png")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~modearte noise~~~~~~~~~~~~~~~~~~~~~~~~~~#
#load the packages
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics, StableRNGs
using ComponentArrays, Lux, Flux, DiffEqFlux, Random, Zygote
using Symbolics, DifferentialEquations, SciMLSensitivity, Plots

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
########solve with nn as UDE#####
#Solve the same problem using NN 
#approximated term kd*q by NN and forming UDE 
##load the packages

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
function predict(θ, X= X= xₙ[:,1], T=t)
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
# Define the output file path
output_file = "UDE_moderate noise_loss.txt"
# Open the file once at the start of the code
file = open(output_file, "w")
# Write the header to the file
println(file, "Iteration, moderate noise Loss")
callback = function(p,l)
    push!(losses_noisy, l)
    # Write the current iteration and loss to the file
    println(file, "$(length(losses_noisy)), $l")
    # Flush the file buffer to make sure the data is saved
    flush(file)
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
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
p_final=opt_sol3.u
nn_ude_pred_noisy_sol=predict(p_final, xₙ[:,1], t)
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms

####################################Symbolic regression####################################
## Start the automatic discovery
@variables t x(t) 
u = [x]
basis = Basis(polynomial_basis(u, 5), u, iv = t)

problem = DirectDataDrivenProblem(nn_ude_pred_noisy_sol, nn_pred_missing_terms)
println(problem)
@show size(problem)
# Create the thresholds which should be used in the search process
λ = exp10.(-3:0.01:5)
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
dddsol2 = solve(problem, basis, opt, maxiter = 10000, progress = true)
println(get_basis(dddsol2))
dddsol2_basis=get_basis(dddsol2)
get_parameter_values(dddsol2_basis)


p=[α, β, γ, δ]
function recovered_ude(du, u, p, t)
    u_rec = dddsol2_basis(u, p)
    du[1]= p_initial[1]*p_initial[3]*(p_initial[4]-u[1])-u_rec[1]
end

estimation_prob = ODEProblem(recovered_ude, u0, tspan, get_parameter_values(dddsol2_basis))
estimate = solve(estimation_prob, Tsit5(), saveat = tsteps)
u_rec = dddsol2_basis(u, get_parameter_values(dddsol2_basis))
# Define the output file path
sr_output_file = "SR_UDE_moderate noise_initial.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
    println(file, "Equation: ", dddsol2_basis)
    println(file, "Parameters: ", get_parameter_values(dddsol2_basis))
    println(file, "Equation with parameters: ", u_rec)
end
################################optimizing the loss###############################
function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_noisy_sol)))
    sum(abs2, nn_pred_missing_terms .- Y)
end

optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(dddsol2_basis))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 10000)

# Look at long term prediction
tspan=(0.0, 350.0)
datasize=50
tsteps=range(tspan[1], tspan[2], length=datasize)
estimation_prob = ODEProblem(recovered_ude, u0, tspan, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = tsteps) # Using higher tolerances here results in exit of julia
u_rec_updated = dddsol2_basis(u, parameter_res)

# Define the output file path
sr_output_file = "SR_UDE_moderate_noise_final.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
# Write the header to the file
println(file, "Equation: ", dddsol2_basis)
println(file, "Updated Parameters: ", parameter_res)
println(file, "Equation with updated parameters: ", u_rec_updated)
end
#####################plotting###########################
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px

ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
plot_true_UDE_approximation=plot(ts, true_sol_noisy[1, :], seriestype=:scatter, marker=:circle, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelcolor=:darkblack, title="Langmuir Adsorption",  size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(ts, transpose(nn_ude_pred_noisy_sol), seriestype=:line, lw=3, label="Predicted data", color=:black)
plot!(ts, estimate_long[1,:], seriestype=:line, linestyle=:dash,   lw=4, label="Regression data", color=:lightgreen)
savefig("01_moderate noise_true data_UDE_reg_pred_data.png")


######PLoting of missing terms#######
# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_noisy_sol[1, :])'
Ȳ=true_missing_terms
println(Ȳ)
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
println(Ŷ)
# compute regression missing terms
reg_pred_missing_terms = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_sol)))
Y=reg_pred_missing_terms
println(Y)

# Plot
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px
plot(ts, true_missing_terms', seriestype=:scatter, marker=:circle, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)",  labelcolor=:black, title="UDE Missing Term", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(ts, nn_pred_missing_terms', seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")
plot!(ts, reg_pred_missing_terms', seriestype=:line, lw=3, color=:blue, label="SR Approximation")
savefig("01_moderate noise_true_UDE_reg_pred_missing_terms.png")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~high noise~~~~~~~~~~~~~~~~~~~~~~~~~~#
#load the packages
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics, StableRNGs
using ComponentArrays, Lux, Flux, DiffEqFlux, Random, Zygote
using Symbolics, DifferentialEquations, SciMLSensitivity, Plots

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
#add high noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-1
xₙ= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_noisy=Array(xₙ)
########solve with nn as UDE#####
#Solve the same problem using NN 
#approximated term kd*q by NN and forming UDE 
##load the packages

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
function predict(θ, X= X= xₙ[:,1], T=t)
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
# Define the output file path
output_file = "UDE_high noise_loss.txt"
# Open the file once at the start of the code
file = open(output_file, "w")
# Write the header to the file
println(file, "Iteration, high noise Loss")
callback = function(p,l)
    push!(losses_noisy, l)
    # Write the current iteration and loss to the file
    println(file, "$(length(losses_noisy)), $l")
    # Flush the file buffer to make sure the data is saved
    flush(file)
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
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
p_final=opt_sol3.u
nn_ude_pred_noisy_sol=predict(p_final, xₙ[:,1], t)
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms

####################################Symbolic regression####################################
## Start the automatic discovery
@variables t x(t) 
u = [x]
basis = Basis(polynomial_basis(u, 5), u, iv = t)

problem = DirectDataDrivenProblem(nn_ude_pred_noisy_sol, nn_pred_missing_terms)
println(problem)
@show size(problem)
# Create the thresholds which should be used in the search process
λ = exp10.(-3:0.01:5)
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
dddsol2 = solve(problem, basis, opt, maxiter = 10000, progress = true)
println(get_basis(dddsol2))
dddsol2_basis=get_basis(dddsol2)
get_parameter_values(dddsol2_basis)


p=[α, β, γ, δ]
function recovered_ude(du, u, p, t)
    u_rec = dddsol2_basis(u, p)
    du[1]= p_initial[1]*p_initial[3]*(p_initial[4]-u[1])-u_rec[1]
end

estimation_prob = ODEProblem(recovered_ude, u0, tspan, get_parameter_values(dddsol2_basis))
estimate = solve(estimation_prob, Tsit5(), saveat = tsteps)
u_rec = dddsol2_basis(u, get_parameter_values(dddsol2_basis))
# Define the output file path
sr_output_file = "SR_UDE_high noise_initial.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
    println(file, "Equation: ", dddsol2_basis)
    println(file, "Parameters: ", get_parameter_values(dddsol2_basis))
    println(file, "Equation with parameters: ", u_rec)
end
################################optimizing the loss###############################
function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_noisy_sol)))
    sum(abs2, nn_pred_missing_terms .- Y)
end

optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(dddsol2_basis))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 10000)

# Look at long term prediction
tspan=(0.0, 350.0)
datasize=50
tsteps=range(tspan[1], tspan[2], length=datasize)
estimation_prob = ODEProblem(recovered_ude, u0, tspan, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = tsteps) # Using higher tolerances here results in exit of julia
u_rec_updated = dddsol2_basis(u, parameter_res)

# Define the output file path
sr_output_file = "SR_UDE_high_noise_final.txt"
# Open the file once at the start of the code
open(sr_output_file, "w") do file
# Write the header to the file
println(file, "Equation: ", dddsol2_basis)
println(file, "Updated Parameters: ", parameter_res)
println(file, "Equation with updated parameters: ", u_rec_updated)
end
#####################plotting###########################
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px
t=sol.t
ts= first(sol.t):(mean(diff(sol.t))):last(sol.t)
plot_true_UDE_approximation=plot(ts, true_sol_noisy[1, :], seriestype=:scatter, marker=:circle, alpha=0.5, color=:blue, label="Training data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)", labelcolor=:darkblack, title="Langmuir Adsorption",  size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(ts, transpose(nn_ude_pred_noisy_sol), seriestype=:line, lw=3, label="Predicted data", color=:black)
plot!(ts, estimate_long[1,:], seriestype=:line, linestyle=:dash,   lw=4, label="Regression data", color=:lightgreen)
savefig("01_high noise_true data_UDE_reg_pred_data.png")


######PLoting of missing terms#######
# Compute the true interactions 
true_missing_terms=p_initial[2].* (nn_ude_pred_noisy_sol[1, :])'
Ȳ=true_missing_terms
println(Ȳ)
# compute the neural network guess of the interactions
nn_pred_missing_terms = nn(nn_ude_pred_noisy_sol, p_final, st)[1]
Ŷ=nn_pred_missing_terms
println(Ŷ)
# compute regression missing terms
reg_pred_missing_terms = reduce(hcat, map(Base.Fix2(dddsol2_basis, p), eachcol(nn_ude_pred_sol)))
Y=reg_pred_missing_terms
println(Y)

# Plot
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 25px
right_margin = 15px
top_margin = 30px
bottom_margin = 25px
plot(ts, true_missing_terms', seriestype=:scatter, marker=:circle, alpha=0.8, color=:red, label="Actual term",xlabel="Time (min)", ylabel="Desorption Rate (mg/g min)",  labelcolor=:black, title="UDE Missing Term", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:topright, grid=false, markersize=10, xlabelfontsize=18, ylabelfontsize=18, titlefontsize=28, xtickfontsize=18, ytickfontsize=18, legendfontsize=18)
plot!(ts, nn_pred_missing_terms', seriestype=:line, lw=3, color=:seagreen, label="UDE Approximation")
plot!(ts, reg_pred_missing_terms', seriestype=:line, lw=3, color=:blue, label="SR Approximation")
savefig("01_high noise_true_UDE_reg_pred_missing_terms.png")