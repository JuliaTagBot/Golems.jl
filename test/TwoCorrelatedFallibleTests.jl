#This model is a work in progress.
#Not currently included in the tests, because any tests that pass it don't deserve to be called tests.
#If you copy and paste this code (while using Golems and Base.Tests) it should run without errors for a little while, but it is cheating.
#The problem is that it does not converge during the maximization step.
#So setting the initial values for the maximization at the truth leads to results magically biased toward the truth.
#It would be reasonable to seed inits with prior data; if the number of iterations is reduced, that would speed things up in general.
#But you can't magincally cheat with real data. If you already know the truth, why are you even running an analysis?
#Anyway, even with all that said, it still errors as is when p = 5.

srand(12345)
pess = 100
S = 0.8 .+ 0.2rand(2)
C = 0.8 .+ 0.2rand(2)
cc = [minimum(S) - prod(S), minimum(C) - prod(C)]

@testset for p ∈ 1:7
  m = Model(Golems.CorrErrors{p});
  π_true = rand(p)
  tv = Golems.construct(Golems.CorrErrors{p,Float64}, copy(π_true), S, C, rand(2) .* cc)
  m = Model(SparseQuadratureGrids.GridContainer(length(tv), 6, SparseQuadratureGrids.GenzKeister), tv, Golems.CorrErrors{p})
  n = gen_data(tv, 200, 75, 50)
  ced = Golems.CorrErrorsData(n..., αS1 = S[1]*pess, βS1 = (1-S[1])*pess, αS2 = S[2]*pess, βS2 = (1-S[2])*pess, αC1 = C[1]*pess, βC1 = (1-C[1])*pess, αC2 = C[2]*pess, βC2 = (1-C[2])*pess)
  jp = JointPosterior(m, ced)
  @testset for i ∈ 1:p
    marg = marginal(jp, x -> x.π[i], Normal)
    @test quantile(marg, 0.005) < π_true[i] < quantile(marg, 0.995)
  end
end
