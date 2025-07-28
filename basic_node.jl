# basic_node.jl

using DifferentialEquations
using Flux
using DiffEqFlux
using Optimisers
using Plots

# -------------------------------
# Step 1: Define the true dynamics (ground truth)
# -------------------------------
function true_dynamics!(du, u, p, t)
    du[1] = 2.0 * u[1]
end

u0 = [1.0]                  # Initial state
tspan = (0.0, 1.0)          # Time span
true_prob = ODEProblem(true_dynamics!, u0, tspan)
true_sol = solve(true_prob, Tsit5(), saveat=0.1)

# Our training data
train_ts = true_sol.t
train_data = hcat(true_sol.u...)  # shape: (state_dim, time_steps)

# -------------------------------
# Step 2: Define the Neural ODE model
# -------------------------------
# A simple neural network layer
dudt = Chain(Dense(1, 50, tanh), Dense(50, 1))

# Neural ODE wrapper
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=0.1)

# -------------------------------
# Step 3: Loss Function
# -------------------------------
function loss()
    pred = n_ode(u0)                  # Output shape: (1, num_timesteps)
    return Flux.Losses.mse(pred, train_data)
end

# -------------------------------
# Step 4: Optimizer
# -------------------------------
opt = Optimisers.Adam(0.05)
ps = Flux.params(dudt)

# -------------------------------
# Step 5: Training Loop
# -------------------------------
epochs = 300

for epoch in 1:epochs
    grads = gradient(() -> loss(), ps)
    Flux.Optimise.update!(opt, ps, grads)
    
    if epoch % 50 == 0 || epoch == 1
        @info "Epoch $epoch, Loss = $(loss())"
    end
end

# -------------------------------
# Step 6: Plot results
# -------------------------------
predicted = n_ode(u0)
plot(train_ts, train_data[1, :], label="True", lw=2)
plot!(train_ts, predicted[1, :], label="Neural ODE", lw=2, linestyle=:dash)
