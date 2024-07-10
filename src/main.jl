
function bolfi!(bolfi::BolfiProblem;
    acquisition::BolfiAcquisition = PostVarAcq(),
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::Union{<:TermCond, <:BolfiTermCond} = IterLimit(1),
    options::BolfiOptions = BolfiOptions(),    
)
    boss_acq = AcqWrapper(acquisition, bolfi, options)
    boss_term_cond = TermCondWrapper(term_cond, bolfi)
    boss_options = create_boss_options(options, bolfi)

    bo!(bolfi.problem;
        acquisition = boss_acq,
        model_fitter,
        acq_maximizer,
        term_cond = boss_term_cond,
        options = boss_options,
    )
    return bolfi
end
