

p = rand(1:100)

x = randn(p)
S = randn(200, p) |> x -> x' * x
Σ = randn(200, p) |> x -> x' * x
B = randn(p,p)
U = chol(S)
Si = inv(S)


@testset begin
  @test all(Golems.symQuad(x, S) .≈ x' * S * x)
  @test Golems.traceProd(S, B) ≈ trace(S * B)
  @test Golems.traceSymProd(S, Σ) ≈ trace(S * Σ)
  @test Golems.traceSymProd(S, Σ) ≈ 2Golems.htraceSymProd(S, Σ)
  @test all(Golems.invSym!(S) .≈ Si)
  Golems.chol!(S)
  @test logdet(U) ≈ -Golems.logTriangleDet(S)
end
