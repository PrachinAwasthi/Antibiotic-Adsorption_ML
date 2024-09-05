#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`Low   noise  solution-  5e-3, High  noise  solution-  5e-2, Very high noise  solution-  5e-1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#add low noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-3
xₙ_low= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_low_noise=Array(xₙ_low)

#add high noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-2
xₙ_high= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_high_noise=Array(xₙ_high)

#add very high noise interms of mean
x̄= mean(true_sol)
noise_magnitude = 5e-1
xₙ_very_high= true_sol.+(noise_magnitude*x̄).*randn(rng, eltype(true_sol), size(true_sol))
true_sol_very_high_noise=Array(xₙ_very_high)

###PLot  the results
using Plots.PlotMeasures
plot_size = (1200, 600)
left_margin = 20px
right_margin = 20px
top_margin = 20px
bottom_margin = 20px

plot(t, true_sol[1, :], seriestype=:scatter, marker=:circle,markersize=7.0, lw=5, color=:blue, label="True data", xlabel="Time (min)", ylabel="Adsorption Capacity (mg/g)",  labelfontsize=14, labelcolor=:darkblack, title="Langmuir Adsorption", titlefontsize=16, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin, legend=:bottomright, grid=false, legendfontsize=14)
plot!(t, true_sol_low_noise[1, :], seriestype=:line,  linestyle=:dash, lw=5, label="True low noise data", color=:red)
plot!(t, true_sol_high_noise[1, :], seriestype=:line,  linestyle=:dash, lw=5, label="True high noise data", color=:orange)
plot!(t, true_sol_very_high_noise[1, :], seriestype=:line,  linestyle=:dash, lw=5, label="True very high noise data", color=:magenta)
savefig("true solution_low  nnoise_high noise_very high noise.png")
