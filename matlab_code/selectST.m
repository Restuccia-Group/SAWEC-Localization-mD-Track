function childSelectST = selectST(child, unperturbed_mat, numST, numSTSelected)

perturbedST = randperm(numST, numSTSelected);
unperturbedST = linspace(1, numST, numST);
unperturbedST(perturbedST) = [];

childSelectST = child;
childSelectST(unperturbedST, :, :) = unperturbed_mat(unperturbedST, :, :);

end