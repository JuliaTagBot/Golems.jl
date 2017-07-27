module Golems

using TensorOperations, StaticArrays, SparseQuadratureGrids, ConstrainedParameters, LogDensities, JointPosteriors, Distributions

import  LogDensities.negative_log_density,
        LogDensities.negative_log_density!,
        LogDensities.Model,
        ConstrainedParameters.construct,
        ConstrainedParameters.type_length,
        ConstrainedParameters.log_jacobian!,
        ConstrainedParameters.update!,
        ConstrainedParameters.Data
#Quick note: LogDensities.negative_log_density looks in Main for the log_density function.


export  TF_RE_ANOVA_Data_balanced,
        TF_RE_ANOVA_Data,
        TF_RE_ANOVA,
        TF_RE_MANOVA_Data_balanced,
        TF_RE_MANOVA_Data,
        TF_RE_MANOVA,
        Golems,
        Model,
        JointPosterior,
        marginal,
        gen_data,
        Normal

include("helper_functions.jl")
include("ANOVA.jl")
include("MANOVA.jl")
include("TwoCorrelatedFallibleTests.jl")



end # module
