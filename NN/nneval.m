function [loss] = nneval(nn, loss, train_x, train_y, val_x, val_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 4 || nargin == 6, 'Wrong number of arguments');

% training performance
nn                    = nnff(nn, train_x, train_y);
loss.train.e(end + 1) = nn.L;

% validation performance
if nargin == 6
    nn                    = nnff(nn, val_x, val_y);
    loss.val.e(end + 1)   = nn.L;
end

end
