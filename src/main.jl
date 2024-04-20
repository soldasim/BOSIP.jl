
function bolfi!(problem::BolfiProblem;
    acquisition::AcquisitionFunction=PDFVariance(),
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),    
)
    bo!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
end
