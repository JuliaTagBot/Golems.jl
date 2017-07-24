
pess = 30
S = 0.8 .+ 0.2rand(2)
C = 0.8 .+ 0.2rand(2)
covsc = min.(S, C) .- S .* C

@testset for p ∈ [1,rand(2:10)]
  m = Model(Golems.CorrErrors{p});
  tv = Golems.construct(Golems.CorrErrors{p,Float64}, 0.5rand(p), S, C, covsc)
  m = Model(Golems.CorrErrors{p})
  n = gen_data(tv, 2000, 750, 500)
  ced = Golems.CorrErrorsData(n..., αS1 = S[1]*pess, βS1 = (1-S[1])*pess, αS2 = S[2]*pess, βS2 = (1-S[2])*pess, αC1 = C[1]*pess, βC1 = (1-C[1])*pess, αC2 = C[2]*pess, βC2 = (1-C[2])*pess)
  jp = JointPosterior(m, ced)
  @testset for i ∈ 1:p
    marg = marginal(jp, x -> x.π[i], Normal)
    @test quantile(marg, 0.0001) < tv.π < quantile(marg, 0.9999)
  end
end
