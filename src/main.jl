
function bolfi!(problem::BolfiProblem;
    acquisition::BolfiAcquisition=PDFVariance(),
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),    
)
    boss_acq = acquisition(problem)

    bo!(problem.problem;
        acquisition=boss_acq,
        model_fitter,
        acq_maximizer,
        term_cond,
        options,
    )
    return problem
end
