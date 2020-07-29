% Resp = G(sigm) - a * G(b * sigm),
function val = DOGAnalysis(a, b, sigma, r,k)
center = [0,0];

[R1,C1] = ndgrid(-15:15, -15:15);
[R2,C2] = ndgrid(-15:15, -15:15);
%Temp = 0.25./ sqrt((R1-center(1)).^2 + (C1-center(2)).^2);
%surf(R1,C1,Temp);
%surf(R1,C1,2.*MAT1 - Temp);

MAT1 = gaussC(R1,C1, sigma, center);

MAT2 = gaussC(R2,C2, (b*sigma), center);

MAT3 = MAT1 - a.*MAT2;

val = MAT3((16-r:16+r),(16-r:16+r));

% figure(k)
% surf(R1,C1,MAT3);

