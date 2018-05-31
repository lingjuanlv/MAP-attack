no_auto=numel(sae.ae);%floor(nn.n/2);
for i=1:no_auto
    nn.W{i}=sae.ae{i}.W{1};
    if nn.size(:,end)~=1 % if Autoencoder add decoder weights
        nn.W{2 * no_auto - i + 1}=sae.ae{i}.W{2};
    end
end