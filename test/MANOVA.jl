using Distributions

srand(123)

function rgauge(Θ::TF_RE_MANOVA)
  ConstrainedParameters.update_U_inverse!(Θ.Σβ);
  ConstrainedParameters.update_U_inverse!(Θ.Στ);
  ConstrainedParameters.update_U_inverse!(Θ.Σθ);
  ConstrainedParameters.update_U_inverse!(Θ.Σϵ);
  ConstrainedParameters.update_Σ!(Θ)
  det(chol(Θ.Στ + Θ.Σθ + Θ.Σϵ)) * ConstrainedParameters.inv_root_det(Θ.Σβ)
end

function gen_data(tv::TF_RE_MANOVA{p,Float64}, μ::Vector{Float64}, b::Int, t::Int, n::Int) where {p}
  bt = b*t
  btn = bt*n
  β_true = rand(MvNormal(convert(Array{Float64,2},tv.Σβ.Σ)), b)
  τ_true = rand(MvNormal(convert(Array{Float64,2},tv.Στ.Σ)), t)
  θ_true = Array{Float64,3}(p, b, t)
  for i ∈ 1:b
    θ_true[:,i,:] = rand(MvNormal(convert(Array{Float64,2},tv.Σθ.Σ)), t)
  end
  yb = Vector{Int64}(btn)
  yt = Vector{Int64}(btn)
  y = Array{Float64,2}(btn, p)
  k = 1:n
  for i ∈ 1:b, j ∈ 1:t
    y[k,:] = rand(MvNormal(μ .+ β_true[:,i] .+ τ_true[:,j] .+ θ_true[:,i,j], convert(Array{Float64,2}, tv.Σϵ)), n)'
    yb[k] .= i
    yt[k] .= j
    k += n
  end
  y, yb, yt
end
function testLogDetandQuad(b::Int, t::Int, n::Int, p::Int, ρ::Real, scales::Vector{<:Real})

  μ = rand(MvNormal(100*eye(p)))
  Σgen = InverseWishart(4, (1-ρ) .* eye(p) .+ ρ .* ones(p,p))
  Θ = Golems.construct(TF_RE_MANOVA{p, Float64}, (scales .* rand(Σgen, 4))... )

  r_true = rgauge(Θ)

  y, yb, yt = gen_data(Θ, μ, b, t, n);

  manova = Model(TF_RE_MANOVA, 2);
  data = TF_RE_MANOVA_Data(y, yb, yt);

  V = kron(ones(b*t*n,b*t*n), data.Σμ) .+ kron(eye(b), kron(ones(t*n,t*n), Θ.Σβ.Σ)) .+ kron(ones(b,b),kron(eye(t),kron(ones(n,n),Θ.Στ.Σ))) .+ kron(eye(b*t),kron(ones(n,n),Θ.Σθ.Σ)) .+ kron(eye(b*t*n),Θ.Σϵ.Σ);

  Vi = inv(V);
  vy = vec(y');

  ConstrainedParameters.update_Σ!(Θ)

  Golems.compΛ!(Θ, data)
  for i ∈ eachindex(data.Λcomp)
    Golems.chol!(data.Λcomp[i])
    Golems.inv!(data.Λcomp[i])
  end
  ldet = Golems.compLogDet(Θ, data)
  quad_p1 = ConstrainedParameters.htrace_AΣinv(data.Γϵ - I, Θ.Σϵ)
  for i ∈ eachindex(data.Λcomp)
    Golems.UUt!(data.Λcomp[i])
  end

  quad_term = quad_p1 + Golems.compQuad(Θ, data)

  jp = JointPosterior(manova, data)
  m = marginal(jp, rgauge)
  @testset begin
    @test quad_term ≈ vy' * Vi * vy / 2
    @test 2ldet - b*t*p*log(n) ≈ logdet(Vi)
    @test quantile(m, 0.025) < r_true < quantile(m, 0.975)
  end

end


b = 25; t = 12; n = 10; p = 2;
ρ = 0.4
scales = [15., 1., 0.4, 0.1]

testLogDetandQuad(b, t, n, p, ρ, scales)
