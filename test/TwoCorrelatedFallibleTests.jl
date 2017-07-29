#This model really isn't very stable.
#change the seed, or increase `p`, and it will probably fail as is.
#maybe it should fail, as this thing needs work.

srand(1)
pess = 100
S = 0.8 .+ 0.2rand(2)
C = 0.8 .+ 0.2rand(2)
cc = [minimum(S) - prod(S), minimum(C) - prod(C)]

@testset for p ∈ 1:7
  m = Model(Golems.CorrErrors{p});
  tv = Golems.construct(Golems.CorrErrors{p,Float64}, rand(p), S, C, (0.5 .+ 0.5 .* rand(2)) .* cc)
  n = gen_data(tv, 200, 75, 50)
  ced = Golems.CorrErrorsData(n..., αS1 = S[1]*pess, βS1 = (1-S[1])*pess, αS2 = S[2]*pess, βS2 = (1-S[2])*pess, αC1 = C[1]*pess, βC1 = (1-C[1])*pess, αC2 = C[2]*pess, βC2 = (1-C[2])*pess)
  jp = JointPosterior(m, ced)
  @testset for i ∈ 1:p
    marg = marginal(jp, x -> x.π[i], Normal)
    @test quantile(marg, 0.005) < tv.π[i] < quantile(marg, 0.995)
  end
end
