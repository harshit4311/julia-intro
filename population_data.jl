# using DifferentialEquations
# using DiffEqFlux
# using Lux
# using Zygote
# using Optimization, OptimizationOptimisers
# using Plots
# using Random


# # 1. Define Lotka-Volterra system to generate synthetic data
# function lotka_volterra!(du, u, p, t)
#     x, y = u
#     α, β, δ, γ = p
#     du[1] = α * x - β * x * y
#     du[2] = δ * x * y - γ * y
# end

# u0 = Float32[1.0, 1.0]
# tspan = (0.0f0, 10.0f0)
# p_real = Float32[1.5, 1.0, 1.0, 3.0]

# prob = ODEProblem(lotka_volterra!, u0, tspan, p_real)
# sol = solve(prob, Tsit5(), saveat=0.1f0)

# X_train = Array(sol)      # (2, T)
# t_train = sol.t           # vector of time points

# # 2. Define neural ODE model using Lux
# dudt_model = Lux.Chain(
#     Lux.Dense(2, 16, tanh),
#     Lux.Dense(16, 2)
# )

# rng = Random.default_rng()
# p_model, st = Lux.setup(rng, dudt_model)

# n_ode = NeuralODE(dudt_model, tspan, Tsit5(), saveat=t_train)

# # 3. Define loss function
# function loss(p_model)
#     pred, _ = n_ode(u0, p_model, st)
#     return Flux.Losses.mse(pred, X_train)
# end

# # 4. Train using Optimization.jl
# optf = OptimizationFunction((x, p) -> loss(x), AutoZygote())
# optprob = OptimizationProblem(optf, p_model)
# res = solve(optprob, ADAM(0.01), maxiters=300)

# # 5. Final prediction
# pred, _ = n_ode(u0, res.u, st)

# # 6. Plot
# plot(t_train, X_train[1, :], label="True Prey", lw=2)
# plot!(t_train, X_train[2, :], label="True Predator", lw=2)
# plot!(t_train, pred[1, :], label="Pred Prey", lw=2, ls=:dash)
# plot!(t_train, pred[2, :], label="Pred Predator", lw=2, ls=:dash)
# xlabel!("Time")
# ylabel!("Population")
# title!("Neural ODE vs True Dynamics")




using DifferentialEquations
using DiffEqFlux
using Flux
using Flux: Chain, Dense, params, train!, ADAM, update!
using Random
using Plots



# 1. Lotka-Volterra equations
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = δ * x * y - γ * y
end

# 2. Generate synthetic data
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0
p_real = [1.5, 1.0, 1.0, 3.0]

prob = ODEProblem(lotka_volterra!, u0, tspan, p_real)
sol = solve(prob, Tsit5(), saveat=tsteps)
X_train = Array(sol)
t_train = sol.t

# 3. Neural network model
nn_model = Chain(
    Dense(2, 16, tanh),
    Dense(16, 2)
)

ps = Flux.params(nn_model)
neural_ode = NeuralODE(nn_model, tspan, Tsit5(), saveat=t_train)

# 4. Loss function
function loss()
    pred = neural_ode(u0)
    return Flux.Losses.mse(pred, X_train)
end

# 5. Train
opt = Flux.ADAM(0.01)
@info "Starting training..."
for epoch in 1:300
    gs = gradient(() -> loss(), ps)
    Flux.Optimise.update!(opt, ps, gs)
    if epoch % 50 == 0
        @info "Epoch $epoch - Loss: $(loss())"
    end
end

# 6. Plot
pred = neural_ode(u0, ps)
plot(t_train, X_train[1, :], label="True Prey", lw=2)
plot!(t_train, X_train[2, :], label="True Predator", lw=2)
plot!(t_train, pred[1, :], label="Pred Prey", lw=2, ls=:dash)
plot!(t_train, pred[2, :], label="Pred Predator", lw=2, ls=:dash)
xlabel!("Time")
ylabel!("Population")
title!("Neural ODE vs True Dynamics")
