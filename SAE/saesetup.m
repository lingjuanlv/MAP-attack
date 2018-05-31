function sae = saesetup(sae,size)
    for u = 2 : numel(size)
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);
        sae.ae{u-1}.learningRate              = sae.learningRate;
        %sae.ae{u-1}.inputZeroMaskedFraction   = .5;
        sae.ae{u-1}.finalMomentum             = .9;
        sae.ae{u-1}.weightPenaltyL2           = sae.weightPenaltyL2;
        sae.ae{u-1}.nonSparsityPenalty        = sae.nonSparsityPenalty;
        sae.ae{u-1}.jacobianPenalty           = sae.jacobianPenalty;
        sae.ae{u-1}.alg                       ='Auto';
    end
end
