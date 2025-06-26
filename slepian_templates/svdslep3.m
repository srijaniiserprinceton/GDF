function varargout=svdslep3(XY,KXY,J,ngro,tol,xver)
% [E,V,c11cmnR,c11cmnK,SE,KXY]=SVDSLEP3(XY,KXY,J,ngro,tol,xver)
%
% Two-dimensional Slepian functions with arbitrary concentration/limiting
% regions in the Cartesian spatial and (half-)spectral domains.
%
% INPUT:
%
% XY       [X(:) Y(:)] coordinates of a SPATIAL-domain curve, in PIXELS
% KXY      [X(:) Y(:)] coordinates of a SPECTRAL-domain HALF-curve, 
%          i.e. in the POSITIVE (including zero) spectral halfplane. 
%          The coordinates here are RELATIVE to the ultimate size of that
%          half-plane, with the CORNER points assuming UNIT wavenumber 
%          values, so they will be ratios, fractions of the Nyquist
%          plane, after the computational growth of the SPACE plane
% J        Number of eigentapers requested [default: 10] 
% ngro     The computational "growth factor" [default: 3]
% tol      abs(log10(tolerance)) for EIGS [default: 12]
% xver     Performs excessive verification, making plots
%
% OUTPUT:
%
% E        The eigenfunctions of the concentration problem
% V        The eigenvalues of the concentration problem
% c11cmnR  The spatial coordinates of the top left corner after growth
% c11cmnK  The spectral coordinates of the top left corner after growth
% SE       The periodogram of the eigenfunctions
% XY       The input spatial-domain curve
% KXY      The symmetrized spectral-space domain curve
%
% EXAMPLE:
%
% svdslep3('demo1',ngro) % with ngro the growth factor
% 
% SEE ALSO:
%
% LOCALIZATION2D
%
% Last modified by fjsimons-at-alum.mit.edu, 08/09/2022

% Default values
defval('J',10);
defval('ngro',1);
defval('tol',12);
defval('xver',0);

% Default SPATIAL curve is a CIRCLE in PIXEL space, of radius cR and cN points
defval('cR',30)
defval('cN',41)
defval('XY',...
       cR*[cos(linspace(0,2*pi,cN)) ; sin(linspace(0,2*pi,cN))]')

if ~isstr(XY)
  % And some (half-)square in the SPECTRAL (half-)space
  % Here you use the notion of the Shannon ratio as in SVDSLEP2, which is
  % relative to the grown area which has unit (kx,ky) in the CORNERS
  defval('R',0.1)
  % Remember this R is strictly for convenience in the next line
  defval('KXY',...
	 R*[-1 1  1 -1 -1; 1 1  0  0  1]')

  % Check if we've already computed these
  % Make a hash with the input variables so you don't recompute
  keyboard
%   fname=hash([XY(:)' KXY(:)' J ngro tol],'SHA-256');
%   % You're going to need an environmental variable and an appropriate directory
%   fnams=fullfile(getenv('IFILES'),'HASHES',sprintf('%s_%s.mat',upper(mfilename),fname));

  % Compute and save or load if presaved
  if 1%~exist(fnams,'file')
    tt=tic;

    % Check the curves and return the range on the inside 
    % For the SPATIAL part, before the growth domain, in pixels
    [XY,xylimt,NyNx]=ccheck(XY,0,[1 1],xver);

    % Now embed this in a larger-size matrix to get rid of edge effects and
    % control the spectral discretization. The spatial domain is the unique
    % reference, in pixels, for everything that comes below. Mind the order!
    newsize=ngro*NyNx;
    
    % Expand the SPATIAL domain and recompute the coordinates
    [QinR,c11cmnR,QX,QY]=cexpand(XY,0,newsize,NyNx,xylimt,xver);

    % For the SPECTRAL part, mirrored, in the discretization appropriate to
    % the growth domain, still relative to the Nyquist plane, not final,
    % only the curve is needed, the expansion works off the spatial grid
    KXY=ccheck(KXY,1,1./newsize,xver*0);

    % Expand the SPECTRAL domain to return the full Nyquist plane in the
    % dimensions of the SPATIAL coordinates, which control the discretization
    [QinK,c11cmnK,QKX,QKY]=cexpand(KXY,1,newsize,[],[],xver);

    % Ensure hermiticity if the domain is even knowing it was square
    if ~any(rem(newsize,2))
      % The dci component (see KNUM2) will be in the lower right quadrant
      dci=[floor(newsize(1)/2)+1 floor(newsize(2)/2)+1];
      % Have not needed to adapt this special case lately
    end
        
    % The result must be Hermitian!
    dom=zeros(newsize); dom(QinK)=1;
    disp(sprintf('\nChecking for Hermitian symmetry\n'))
    difer(isreal(ifft2(ifftshift(dom)))-1,[],1,[])

    % Now you are ready for a check, really a repeat of what's been CCHECKED
    % and CEXPAND already
    if xver==1
      figure(3)
      clf
      % Plot the SPATIAL region of interest exactly as input
      ah(1)=subplot(221);
      twoplot(XY,'b','LineWidth',1.5); hold on
      % Compute the centroid - my function is from SAGA, but R2022a has its own
      [X0,Y0]=centroid(XY(:,1),XY(:,2));
      plot(X0,Y0,'b+')
      axis equal; grid on
      % Just open up the axes a bit
      xlim(minmax(XY(:))*1.25)
      ylim(minmax(XY(:))*1.25)
      t(1)=title('Space-domain input curve and its centroid');
      xlabel('horizontal pixels')
      ylabel('vertical pixels')

      % Plot the SPECTRAL region of interest after the mirroring operation
      ah(2)=subplot(222);
      twoplot([KXY ; KXY(1,:)],'r','LineWidth',1.5); hold on
      % This centroid remains zero
      [KX0,KY0]=deal(0);
      plot(KX0,KY0,'ro')
      axis equal; grid on
      % Open up the axes to the full Nyquist plane, corners at [+/-1 +/-1]
      axis([-1 1 -1 1])
      t(2)=title('Spectral-domain input curve');
      xlabel('scaled horizontal wavenumbers')
      ylabel('scaled vertical wavenumbers')

      ah(3)=subplot(223);
      % Plot the SPATIAL region of interest after the growth domain
      plot(QX(QinR),QY(QinR),'b.')
      % The original curve in PIXEL coordinates needs to plot right on here
      hold on
      twoplot(XY,'y','LineWidth',2)
      hold off
      axis equal; grid on
      xlim(minmax(QX(:))*1.25)
      ylim(minmax(QY(:))*1.25)
      t(3)=title(sprintf('Space-domain region of interest (ngro %i)',ngro));
      xlabel('horizontal pixels')
      ylabel('vertical pixels')

      ah(4)=subplot(224);
      % Plot the SPECTRAL region of interest after the growth domain
      plot(QKX(QinK),QKY(QinK),'r.')
      % The curve in FRACTIONAL coordinates needs to plot right on here
      hold on
      twoplot([KXY ; KXY(1,:)],'y','LineWidth',2)
      hold off
      grid on
      axis([-1 1 -1 1])
      t(3)=title('Spectral-domain region of interest');
      xlabel('scaled horizontal wavenumbers')
      ylabel('scaled vertical wavenumbers')

      ntix=4;
      % Non-pathological PIXEL coordinates should round gracefully
      set(ah(3),'xtick',sort(unique(round(...
       		 [c11cmnR(1):round(range(c11cmnR([1 3]))/ntix):c11cmnR(3) c11cmnR(3)]))))
      set(ah(3),'ytick',sort(unique(round(...
       		 [c11cmnR(4):round(range(c11cmnR([2 4]))/ntix):c11cmnR(2) c11cmnR(2)]))))
      % SPECTRAL coordinates are already fractions and need to include zero
%      set(ah(4),'xtick',sort(unique(round(...
%       		 [0 c11cmnK(1):round(range(c11cmnK([1 3])*100)/ntix)/100:c11cmnK(3) c11cmnK(3)]*100)/100)))
      set(ah(4),'ytick',sort(unique(round(...
       		 [0 c11cmnK(4):round(range(c11cmnK([2 4]))*100/ntix)/100:c11cmnK(2) c11cmnK(2)]*100)/100)))
      % The above wasn't super cool
      set(ah(4),'xtick',[-1:0.5:1],'ytick',[-1:0.5:1])

      set(ah(3:4),'GridLineStyle',':')

      disp(sprintf('\nType DBCONT to proceed or DBQUIT to quit\n'))
      keyboard
    end

    % The SPECTRAL domain this needs to be turned into a Fourier operator
    QinK=indeks(fftshift(v2s(1:prod(newsize))),QinK);

    % Now make the operators that we are trying to diagonalize
    P=@(x) proj(x,QinR);
    % We're finding VECTORS that are going to be 2-D images!
    Q= @(x) fft2vec(x);
    Qi=@(y) ifft2vec(y);
    L=@(y) proj(y,QinK);
    H=@(x) P(Qi(L(Q(P(x)))));

    % And then find the eigenvectors and eigenvalues
    OPTS.isreal=false;
    OPTS.disp=1;
    defval('tolerance',10^-tol);
    OPTS.tol=tolerance;
    OPTS.maxit=500;

    % Remember to specify the output size
    [E,V]=eigs(H,prod(newsize),J,'LR',OPTS);
    
    [V,i]=sort(diag(V),'descend');
    E=E(:,i); V=V(1:J); E=E(:,1:J);

    % Define some kind of tolerance level
    tol=sqrt(eps); 

    % Make them real as we know they should be
    if any(imag(V)>tol)
      error('Complex eigenvalues');
    else
      V=real(V);
      % Check imaginary part of the "good" eigenfunctions
      disp(sprintf('mean(abs(imag(E))) = %8.3e out of %8.3e\n',...
		   mean(mean(abs(imag(E(:,V>tol))))),...
		   mean(mean(abs(E(:,V>tol))))))
      % Note that they were normalized in the complex plane
      E=real(E); E=E./repmat(diag(sqrt(E'*E))',size(E,1),1);
    end

    % Protect against NaNs
    h=isnan(V); V=V(~h); E=E(:,~h); J=sum(~h);

    if nargout>4
      % Get the power spectrum
      SE=zeros(prod(newsize),size(E,2));
      for i=1:size(E,2)
	SE(:,i)=indeks((abs(fft2(v2s(E(:,i)))).^2),':');
      end
    else
      SE=NaN;
    end

    disp(sprintf('%s took %f seconds',upper(mfilename),toc(tt)))
%     save(fnams,'E','V','c11cmnR','c11cmnK','SE','XY','KXY')
  else
    disp(sprintf('%s loading %s',upper(mfilename),fnams))
%     load(fnams)
  end
  % Output
  varns={E,V,c11cmnR,c11cmnK,SE,XY,KXY};
  varargout=varns(1:nargout);
elseif strcmp(XY,'demo1')
  % Fake the second input now as the growth factor
  defval('KXY',[]); ngro=KXY; clear KXY

  % Randomize the test
  if 0%round(rand)
    % A circle in SPACE...
    cR=30;
    cN=41;
    XY=cR*[cos(linspace(0,2*pi,cN)) ; sin(linspace(0,2*pi,cN))]';
  else
    % A random blob, fix the radius to be something sizable in pixels
    [x,y]=blob(1,2); XY=[x y]*20; 
  end
  if 0 %round(rand)
    % And a BOX in SPECTRAL space, no need to close it as it will get
    % mirrored anyway about the lower symmetry axis...
    R=0.13;
    KXY=R*[-1 -1 1 1 ; 0 1 1 0]';
  else
    % A random blob in relative coordinates that are somewhat appropriate
    [kx,ky]=blob(1,2); KXY=[kx ky]/3; KXY=KXY(KXY(:,2)>=0,:);
    keyboard
  end

  % How many eigenfunctions?
  J=30;
  % Compute the eigenfunctions
  [E,V,c11cmnR,c11cmnK,SE,XY,KXY]=svdslep3(XY,KXY,J,ngro);

  % Plot the first offs+3 basis functions
  offs=0;

  % Make the figures
  figure(1)
  clf
  [ah,ha]=krijetem(subnum(2,3));
  for ind=1:3
    % SPACE-domain functions in PIXEL units
    axes(ah(ind))
    imagefnan(c11cmnR(1:2),c11cmnR(3:4),v2s(E(:,ind+offs)))
    hold on
    plot(XY(:,1),XY(:,2),'b','LineWidth',1.5); hold off
    title(sprintf('%s = %i | %s = %7.5f','\alpha',ind+offs,'\lambda',V(ind+offs)))
    xlabel('horizontal pixels')
    ylabel('vertical pixels')

    % SPECTRAL-domain functions, periodogram
    axes(ha(2*ind))
    psdens=fftshift(decibel(v2s(SE(:,ind+offs))));
    psdens(psdens<-80)=NaN;
    imagefnan(c11cmnK(1:2),c11cmnK(3:4),psdens);
    hold on
    % Remember the original curve was relative to the Nyquist plane
    twoplot([KXY ; KXY(1,:)],'r','LineWidth',1.5); hold off
    xlabel('scaled horizontal wavenumbers')
    ylabel('scaled vertical wavenumbers')
  end

  % Also try this one here
  figure(2)
  clf
  EE=nansum(repmat(V(:)',length(E),1).*E.^2,2);
  SEE=nansum(repmat(V(:)',length(E),1).*SE.^2,2);

  % SPACE-domain functions in PIXEL units
  subplot(121)
  imagefnan(c11cmnR(1:2),c11cmnR(3:4),v2s(EE)); axis image 
  hold on; plot(XY(:,1),XY(:,2),'b','LineWidth',1.5); hold off
  title('Eigenvalue weighted SPATIAL sum')
  xlabel('horizontal pixels')
  ylabel('vertical pixels')

  % SPECTRAL-domain functions, periodogram
  subplot(122)
  psdens=fftshift(decibel(v2s(SEE)));
  psdens(psdens<-80)=NaN;
  imagefnan(c11cmnK(1:2),c11cmnK(3:4),psdens); axis image
  hold on
  % Remember the original curve was relative to the Nyquist plane
  twoplot([KXY ; KXY(1,:)],'r','LineWidth',1.5); hold off
  title('Eigenvalue weighted SPECTRAL sum')
  xlabel('scaled horizontal wavenumbers')
  ylabel('scaled vertical wavenumbers')
  set(gca,'xtick',[-1:0.5:1],'ytick',[-1:0.5:1])


  % Also try this one here
  figure(3)
  clf
  plot(V,'ko-')
  title(sprintf('sum of the eigenvalues %8.3f',sum(V)))
  longticks(gca,2)
  ylim([-0.1 1.1])
  grid on
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=ccheck(XY,isk,qdydx,xver)
% Given PIXEL coordinates of a closed curve XY, makes a symmetric centered
% enclosing grid. For the SPATIAL domain (isk=0), that's it (qdydx=1). For
% the SPECTRAL domain (isk=1), it assumes the curve is in the half-plane,
% and mirrors it, but you now require qdydx. Stripped from LOCALIZATION2D.
%
% OUTPUT:
%
% XY              The input curve (isk=0) or its mirrored version (isk=1)
% [xlimt ylimt]   The coordinate limits of the grid
% [Ny Nx]         The dimensions of the grid

% In the spectral domain the input curve will be mirrored
defval('isk',0)
% This has been uber-verified as it is
defval('xver',0);

% Make sure the XY of the curve has two columns
if ~isempty(find(size(XY,2)==2))
  if size(XY,1)<size(XY,2)
    XY=XY';
  end
else
  error('Coordinates of the curve not as expected')
end

if isk==0
  % The spacings are in pixels and shouldn't have to be input
  qdydx=[1 1];
elseif isk==1
  %  Mirror the curve; do not stick in NaNs or they might end up
  % not matching any input zeros; may need to watch POLY2CW
  XY=[XY ; -XY];
  % The spacings are in 1/pixels but did need to be input
end

if nargout>1 || xver==1
  % Find limits in x and y so as to contain the curves
  xlimt=minmax(XY(:,1));
  ylimt=minmax(XY(:,2));
  
  % You'd use these if you wanted a rectangular grid
  Ny=round([ylimt(2)-ylimt(1)]/qdydx(1));
  Nx=round([xlimt(2)-xlimt(1)]/qdydx(2));
  % ...but we make the dimensions (not the domain!) square for now which helps
  % with the Hermiticity constraint and the sampling of the Nyquist plane
  [Nx,Ny]=deal(max(Nx,Ny));
  % Force the output to be ODD for Hermiticity, so the zero-wavenumber is centered
  Nx=Nx+~rem(Nx,2);
  Ny=Ny+~rem(Ny,2);
else
  [xlimt,ylimt,Ny,Nx]=deal(NaN);
end

% Only need to do this if you want to inspect it, it does not get further used
if xver==1
  % Don't be a fanatic about half pixels as in LOCALIZATION2D but 
  % simply strive for symmetry
  qx=linspace(xlimt(1),xlimt(2),Nx);
  qy=linspace(ylimt(1),ylimt(2),Ny);
  
  % Remember that these may not be square depending on the choices above
  [QX,QY]=meshgrid(qx,qy);

  % The "curve" is not the boundary but rather the last set of "elements" on the "grid".
  % The midpoint indices of this subset that fall inside of the region...
  Qin=find(inpolygon(QX,QY,XY(:,1),XY(:,2)));
  
  % Make a plot
  figure(1)
  cplot(XY,QX,QY,Qin,isk,'CCHECK')
end

% Optional output
varns={XY,[xlimt ylimt],[Ny Nx]};
varargout=varns(1:nargout);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Qin,c11cmn,QX,QY]=cexpand(XY,isk,newsize,oldsize,xylimt,xver)
% Expands a rectangular area enclosing a curve to a new size and computes
% the indices of the interior points and the axis limits

defval('isk',0)
% This also should work like a charm all the time
defval('xver',1);

if isk==0
  % How many rows and columns to add on either size?
  addon=round([newsize-oldsize]/2);
  % Actually add to the space domain in pixel spacing
  addx=range(xylimt(1:2))/oldsize(2)*addon(2);
  addy=range(xylimt(3:4))/oldsize(1)*addon(1);
  c11=[xylimt(1) xylimt(4)]+[-addx  addy];
  cmn=[xylimt(2) xylimt(3)]+[ addx -addy];
else
  % The Nyquist plane in wavenumber space in relative coordinates, see KNUM2
  % That's all you need to do, there is no "expansion" to speak of since
  % all we do is work on the discretization of the SPATIAL coordinates.
  % We this note this is only [-1 -1] in the top LEFT corner for EVEN sizes
  c11=2*[-floor( newsize(2)   /2)/newsize(2)  -floor( newsize(1)   /2)/newsize(1)];
  cmn=2*[ floor((newsize(2)-1)/2)/newsize(2)   floor((newsize(1)-1)/2)/newsize(1)];
end
c11cmn=[c11 cmn];

% Now compute the coordinates in the embedding
qx=linspace(c11(1),cmn(1),newsize(2));
qy=linspace(c11(2),cmn(2),newsize(1));
[QX,QY]=meshgrid(qx,qy);
Qin=find(inpolygon(QX,QY,XY(:,1),XY(:,2)));

if xver==1
  % Make a plot
  figure(2)
  cplot(XY,QX,QY,Qin,isk,'CEXPAND')
end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Pv=proj(v,p)
% Projects the vector v on the indices p
Pv=zeros(size(v));
Pv(p)=v(p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fv=fft2vec(v)
% Returns the two-dimensional FFT of a vector
Fv=fft2(v2s(v));
Fv=Fv(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iFv=ifft2vec(Fv)
% Returns the two-dimensional IFFT of a vector
iFv=ifft2(v2s(Fv));
iFv=iFv(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cplot(XY,QX,QY,Qin,isk,cid)
clf
plot(QX,QY,'.','Color',grey)
hold on; axis image
if isk==0
  plot(QX(Qin),QY(Qin),'bo')
else
  plot(QX(Qin),QY(Qin),'ro')
end
hold off
xlim(minmax(QX(:))*1.1)
ylim(minmax(QY(:))*1.1)
shrink(gca,1.25,1.25); longticks(gca,2)
t=title(sprintf('Verifying %s %i x %i isk %i',...
		upper(cid),size(QX),isk));
movev(t,max(abs(QY(:)))/20)
disp(sprintf('\nHit ENTER to proceed or CTRL-C to abort\n'))
% Plot the original curve in actual coordinates
hold on
twoplot([XY ; XY(1,:)],'y','LineWidth',2)
hold off
if isk==0
  xlabel('horizontal pixels')
  ylabel('vertical pixels')
else
  xlabel('scaled horizontal wavenumbers')
  ylabel('scaled vertical wavenumbers')
end
pause

