using DifferentialEquations
using Plots

function unscaledSI!(du,u,q,t)
    p = q[1]
    r = q[2]
    k = q[3]
    β1 = q[4]
    α = q[5]
    μS = q[6]
    μI = q[7]
    β2 = q[8]

    JS1,JI1,AS1,AI1 = u
    # JS1,JI1,AS1,AI1,JS2,JI2,AS2,AI2 = u
    N1 = JS1+JI1+AS1+AI1
    # N2 = JS2+JI2+AS2+AI2
    

    du[1] = p*r*(1-N1/k)*AS1-β1*JS1*(JI1+AI1)/N1-α*JS1-μS*JS1
    du[2] = β1*JS1*(JI1+AI1)/N1-μI*JI1
    du[3] = α*JS1-β1*AS1*(JI1+AI1)/N1-μS*AS1
    du[4] = β1*AS1*(JI1+AI1)/N1-μI*AI1

    # du[5] = (1-p)*r*(1-N1/k)*AS2+p*r*AS2-β2*JS2*(JI2+AI2)/N2-α*JS2-μS*JS2
    # du[6] = β2*JS2*(JI2+AI2)/N2-μI*JI2
    # du[7] = α*JS2-β2*AS2*(JI2+AI2)/N2-μS*AS2
    # du[8] = β2*AS2*(JI2+AI2)/N2-μI*AI2
end

tspan = (0., 100.)
u0 = rand(4)

println(u0)

p = rand()
r = 1
k = 1
β1 = 0.3
α = 1
μS = 0.1
μI = 0.1
β2 = 0.8
q = [p, r, k, β1, α, μS, μI, β2]

prob = ODEProblem(unscaledSI!, u0, tspan, q)
sol = solve(prob,alg_hints=[:stiff])
plot(sol)
savefig("./juliaplot.png")


# using Pkg
# Pkg.add("Plots")