clear;clc;

%**************Claimed Parameters********************************
sigma_E = 1.5; %0.4:0.02:1; %0.4:0.02:1.5;     %distributed excitation
sigma_I = 5;%1.6:0.08:4; %1.6:0.08:6;
FrameSTRT_Num = 1;
FrameEnd_Num = 25;
Max_delay = 2;
LGMD_OutputS = zeros(length(sigma_E),FrameEnd_Num-Max_delay-1);
FFI = zeros(length(sigma_E),FrameEnd_Num-Max_delay-1);
Threshold_Pnt = zeros(length(sigma_E),1);
kernalG = ones(4,4);

a = 1.5;%1.5~1.8  
% b = 4;
r = 6;
Thresh_G_0 = 0.5;
alfa = -0.1;%-0.1,-0.6;
beta = 0.5;%0.5,0.4;
lamda = 0.7;%0.7,0.6;
%****************define h(t)***************************************
x = -r:1:r;
y = -r:1:r;
for i= 1:(r*2+1)
    for j = 1:(r*2+1)
        ht(i,j)= alfa + 1./(beta+ exp(-((x(i)*lamda).^2+(y(j)*lamda).^2)));
    end
end
ht(ht<1) =0;
%***************D-LGMD************************************
% for k = 1:1
for k = 1:length(sigma_E)
    for h = 1:length(sigma_I)
%*********Classic LGMD **********
%     kernel_I = [0.3535 0.4472 0.5 0.4472 0.3535; 
%                 0.4472 0.5 1 0.5 0.4472
%                 0.5 1 0 1 0.5 
%                 0.4472 0.5 1 0.5 0.4472
%                 0.3535 0.4472 0.5 0.4472 0.3535];
% %     kernel_I = [0.125 0.25 0.125
% %                 0.25 0 0.25
% %                 0.125 0.25 0.125];
%     kernel_I = a.* kernel_I
%     kernel_E = 1;       
%**********distributed excitation and inhibition kernels***********    
    Tempkernel = DOGAnalysis(a, sigma_I(h)/sigma_E(k) , sigma_E(k), r ,k);  %Func(a,b,sigma,r);
    kernel_E = GaussAnalysis(sigma_E(k),r);
    kernel_E(ht<1) = Tempkernel(ht<1);   %To form distributed time delay, which is determined by the h(t) distribution.(Self-inhibition involved here)
    kernel_E(ht>1) = 0
    kernel_E(kernel_E<0) = 0;
    Showkerne_E = round (kernel_E*100);

    kernel_I = a * GaussAnalysis(sigma_I(h),r);
    kernel_I(ht<1) =0;%(here I make the h(t) distribution in this principle: delay=0 when ht<1, otherwise delay=1)
    kernel_I_delay1 = kernel_I;

    kernel_I_delay1(ht<1) =0;
    kernel_I_delay1(ht>1.8999) =0
    Showkernel_I_delay1 = round (kernel_I_delay1*100);
    
    kernel_I_delay2 = kernel_I;
    kernel_I_delay2(ht<1.8999) =0
    Showkernel_I_delay2 = round (kernel_I_delay2*100);

%*****************************************************  
 FilePath = 'E:\Simple_Figures\TestImage_';
%***************************************************
tic;
t1 = toc;
    for i=FrameSTRT_Num:FrameEnd_Num-Max_delay-1
        FileName = strcat (FilePath,num2str(i),'.bmp');
        Frame1 = imread(FileName);
        Frame1 = im2single (Frame1);
        FileName = strcat (FilePath,num2str(i+1),'.bmp');
        Frame2 = imread(FileName);
        Frame2 = im2single (Frame2);
        FileName = strcat (FilePath,num2str(i+2),'.bmp');
        Frame3 = imread(FileName);
        Frame3 = im2single (Frame3);
        FileName = strcat (FilePath,num2str(i+3),'.bmp');
        Frame4 = imread(FileName);
        Frame4 = im2single (Frame4);
        
        Frame_Diff1 = abs(Frame2 - Frame1);
        Frame_Diff2 = abs(Frame3 - Frame2);
        Frame_Diff3 = abs(Frame4 - Frame3);

       FFI(i) = sum(sum(Frame_Diff1));
       Thresh_G(i) = FFI(k,i)*0.001/200 * Thresh_G_0;
       Thresh_G(i) = min(Thresh_G(i),0.3);
       
%       (Note: Layer_I = currentframe (conv) kernel_I_delay0 + delayed_1frame (conv)
%       kernel_I_delay1 +delayed_2frame (conv) kernel_I_delay2)
        Layer_E = conv2(Frame_Diff3,kernel_E,'same');
        Layer_I_delay1 = conv2(Frame_Diff2,kernel_I_delay1,'same');
        Layer_I_delay2 = conv2(Frame_Diff1,kernel_I_delay2,'same');
        Layer_I = Layer_I_delay1 + Layer_I_delay2;  %delay0 has been involved to kernal E.
        
       Layer_S = Layer_E - Layer_I;
       Layer_S(Layer_S<0) =0;      %results will not be negative.
% %*************FFM-GD***************
       Layer_G_Cef = conv2(Layer_S,kernalG,'same');
       Layer_G = Layer_S .* Layer_G_Cef;

        Layer_G(Layer_G<Thresh_G(i))=0;
        Layer_G(Layer_G>1)=1;
        LGMD_OutputS(k,i) = sum(sum(Layer_S));    %**output of G or S layer***
        LGMD_OutputG(k,i) = sum(sum(Layer_G));    %**output of G or S layer***
        
        %**************Spike Frequency Adaption*****************
        %         Kt(i) = LGMD_OutputG(k,i);
        Kt(i) = 1/( 1 + exp (-LGMD_G_out(i)/(78000*0.6))); 
        if i >2
            q = q + 1;
            U(q) =  Kt(i) - Kt(i-1);
            U_2(q) =  Kt(i) - 2*Kt(i-1) + Kt(i-2);
            
            if U(q)>= 0.01
                MP(i) = 0.8*Kt(i);
            else
                MP(i) = 0.8 * MP(i-1) + 0.8*U(q);
            end
        end

   figure(1)
   Video_Resize = imresize(Layer_G,0.5);
   imshow(Video_Resize);
    end
    %*************************************************
    Normalized_OutS(k,1:i) = mapminmax(LGMD_OutputS(k,:), 0, 1);
    Normalized_OutG(k,1:i) = mapminmax(LGMD_OutputG(k,:), 0, 1);
    end
end
t2 = toc;
RunTime = t2-t1
%*****************plots********************************************************************************
for k = 1:length(sigma_E)
    figure (3)
    hold on
    plot (1:i,Normalized_OutS(k,(1:i)));
    plot (1:i,Normalized_OutG(k,(1:i)));
end


