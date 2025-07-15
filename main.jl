using Statistics, Distributions, Plots

# Generate random data
data = rand(Normal(100, 15), 1000)

# Compute statistics
println("Mean: ", mean(data))
println("Standard Deviation: ", std(data))

# Plot histogram
histogram(data, bins=30, title="Normal Distribution", xlabel="Value", ylabel="Frequency")
gui()  # force show plot on mac os