using DifferentialEquations
using DiffEqFlux
using Flux
using Plots
using gradient

# 1. Define ODE dynamics (Lotka-Volterra)
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

# 2. Initial condition & time setup
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# 3. Generate "true" data to fit (with true parameters)
true_p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra!, u0, tspan, true_p)
sol = solve(prob, Tsit5(), saveat=tsteps)
observed_data = Array(sol)

# 4. Initial guess for parameters (to optimize)
mutable_p = [2.2, 1.0, 2.0, 0.4]  # must be mutable for training

# 5. Prediction function
function predict(p)
    _prob = remake(prob, p=p)
    Array(solve(_prob, Tsit5(), saveat=tsteps))
end

# 6. Loss: MSE between prediction and true data
loss(p) = sum(abs2, predict(p) .- observed_data)

# 7. Optimizer setup (ADAM)
opt = Flux.Optimiser(Adam(0.1))
state = Flux.setup(opt, mutable_p)

# 8. Training loop (explicit gradient)
for epoch in 1:100
    grads = gradient(mutable_p -> loss(mutable_p), mutable_p)
    Flux.update!(state, mutable_p, grads)
    
    if epoch % 10 == 0
        println("Epoch $epoch | Loss: ", loss(mutable_p))
        display(plot(solve(remake(prob, p=mutable_p), Tsit5(), saveat=tsteps), ylim=(0,6), label=["x" "y"]))
    end
end
