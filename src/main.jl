
function bolfi!(bolfi::BolfiProblem;
    acquisition::BolfiAcquisition=PDFVariance(),
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),    
)
    boss_acq = AcqWrapper(acquisition, bolfi)
    boss_term_cond = TermCondWrapper(term_cond, bolfi)

    bo!(bolfi.problem;
        acquisition=boss_acq,
        model_fitter,
        acq_maximizer,
        term_cond=boss_term_cond,
        options,
    )
    return bolfi
end
