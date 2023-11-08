-- Gkyl ------------------------------------------------------------------------
--
-- We use an analytical Solovev equilibrium and 
-- generate a field-aligned coordinate mapping.
-- For details, see Section 5.3 of Mandell's thesis.
--
--------------------------------------------------------------------------------
local Plasma    = (require "App.PlasmaOnCartGrid").Gyrokinetic()
local Constants = require "Lib.Constants"
local Mpi       = require "Comm.Mpi"
local math      = require("sci.math").generic
local root      = require("sci.root")
local quad      = require("sci.quad")
local diff      = require("sci.diff-recursive")
local df        = diff.df

-- Universal constant parameters.
eps0         = Constants.EPSILON0
eV           = Constants.ELEMENTARY_CHARGE
qe           = -eV
qi           = eV
me           = Constants.ELECTRON_MASS

-- Plasma parameters.
mi           = 2.014*Constants.PROTON_MASS -- (deuterium ions)
Te0          = 40*eV 
Ti0          = 40*eV 
n0           = 7e18  -- [1/m^3]

-- Geometry and magnetic field.
R0 = 2.5 -- [m]
B0           = 0.55 -- [T]
Lpol         = 2.4 -- [m]
Ztop = 6.0


-- Parameters for collisions.
nuFrac = 0.1
-- Electron collision freq.
logLambdaElc = 6.6 - 0.5*math.log(n0/1e20) + 1.5*math.log(Te0/eV)
nuElc = nuFrac*logLambdaElc*eV^4*n0/(6*math.sqrt(2)*math.pi^(3/2)*eps0^2*math.sqrt(me)*(Te0)^(3/2))
-- Ion collision freq.
logLambdaIon = 6.6 - 0.5*math.log(n0/1e20) + 1.5*math.log(Ti0/eV)
nuIon = nuFrac*logLambdaIon*eV^4*n0/(12*math.pi^(3/2)*eps0^2*math.sqrt(mi)*(Ti0)^(3/2))
-- Electron-Ion collision frequency.
nuElcIon     = nuElc/1.96
nuIonElc     = me*nuElcIon/mi -- Ion-electron collision frequency.

-- Derived parameters
vti      = math.sqrt(Ti0/mi)
vte  	 = math.sqrt(Te0/me)
c_s      = math.sqrt(Te0/mi)
omega_ci = math.abs(qi*B0/mi)
rho_s    = c_s/omega_ci


psi_prefactor = 1.0
local function Psi(R, Z)
--  return B0*k/(2*R0^2*q0)*(R^2*Z^2/k^2 + (R^2 - R0^2)^2/4)
   local x = R/R0
   local y = Z/R0
   return psi_prefactor*(0.00373804283369699*x^6*math.log(x) - 0.00574955335438162*x^6 - 0.0448565140043639*x^4*y^2*math.log(x) + 0.0503044260840946*x^4*y^2 + 0.017623348727471*x^4*math.log(x) + 0.0956643504553683*x^4 + 0.0299043426695759*x^2*y^4*math.log(x) - 0.0160920841654771*x^2*y^4 - 0.0704933949098842*x^2*y^2*math.log(x) + 0.0644725519961135*x^2*y^2 - 7.00898484784405e-5*x^2*math.log(x) - 0.303766642191745*x^2 - 0.00199362284463839*y^6 + 0.0117488991516474*y^4 + 7.00898484784405e-5*y^2 + 0.0145368720253975)
end


-- stopping criteria for root-finding
local function stop(tol)
   return function(x, y, xl, xu, yl, yu)
      if diff.lt(math.abs(y), tol) then return true
      else return false
      end
   end
end

local numRcalls = 0
local numMapCalls = 0
-- numerically invert Psi to get R(Z, Psi)
local function R(Z, Psi0)
   -- solve f(Ri) = Psi(Ri,Z) - Psi0 = 0 for Ri
   numRcalls = numRcalls+1
   local function f(Ri) 
      return Psi(Ri,Z)-Psi0
   end
   local val = root.ridders(f, 0.7*R0, 3*R0, stop(1e-12))
   return val
   
   -- for testing, just use the analytical expression because it is less expensive
   --return math.sqrt(R0^2 + (2*(-Z^2 + math.sqrt(2*k^3*Psi0*q0*R0^2 - B0*k^2*R0^2*Z^2 + B0*Z^4)/math.sqrt(B0)))/k^2)
end

local function dRdZ(Z, Psi)
   return df(R,1)(Z,Psi)
end

-- compute normalization factor for equal-arc-length poloidal angle
local function norm(Psi)
   local function integrand(t)
      return math.sqrt(1+dRdZ(t, Psi)^2)
   end
   return quad.dblexp(integrand, -Ztop, Ztop, 1e-15)/(2*math.pi)
end

-- compute poloidal angle coordinate theta
local function theta(R,Z)
   local Psi = Psi(R,Z)
   local function integrand(t)
      return math.sqrt(1+dRdZ(t, Psi)^2)
   end
   local integral, _ = quad.dblexp(integrand, -Ztop, Z, 1e-15)
   return 1/norm(Psi)*integral-math.pi
end

-- numerically invert Psi(R,Z) and theta(R,Z) via root finding
local function RZ(Psi, theta0)
   while diff.lt(theta0, -math.pi) do
      theta0 = theta0+2*math.pi
   end
   while diff.lt(math.pi, theta0) do
      theta0 = theta0-2*math.pi
   end
   local function f(Z)
      local R = R(Z,Psi)
      local theta = theta(R,Z)
      return theta-theta0
   end
   local Zval = root.ridders(f, -Ztop, Ztop, stop(1e-7))
   local Rval = R(Zval, Psi)
   return Rval, Zval
end

local function Bphi(R)
   return B0*R0/R
end

local function grad_Psi(R, Z)
   return df(Psi,1)(R,Z), df(Psi,2)(R,Z), 0
end

local function grad_theta(R, Z)
   return df(theta,1)(R,Z), df(theta,2)(R,Z), 0
end

-- compute alpha (binormal) coordinate
local function alpha(Ri, Zi, phi)
   local Psi = Psi(Ri,Zi)
   local function integrand(t)
      local Rt = R(t, Psi) -- need to parametrize Z with R=t for integral
      local gradPsi_R, gradPsi_Z, _ = grad_Psi(Rt,t)
      local gradPsi = math.sqrt(gradPsi_R^2 + gradPsi_Z^2)
      return math.sqrt(1+dRdZ(t, Psi)^2)/Rt/gradPsi
   end
   local integral 
   if diff.le(1e-14, Zi) then
      integral, _ = quad.dblexp(integrand, 1e-14, Zi, 1e-14)
   else
      integral, _ = -quad.dblexp(integrand, Zi, 1e-14, 1e-14)
   end
   return phi - Bphi(Ri)*Ri*integral
end

local function grad_alpha(R, Z, phi)
   return df(alpha,1)(R,Z,phi), df(alpha,2)(R,Z,phi), df(alpha,3)(R,Z,phi)/R
end

local function jacob_PsiThetaPhi(R,Z)
   local gradPsi_R, gradPsi_Z, _ = grad_Psi(R,Z)
   return norm(Psi(R,Z))*R/math.sqrt(gradPsi_R^2 + gradPsi_Z^2)
end

-- invert mappings via root finding
local function RZphi(Psi, theta0, alpha0)
   while diff.lt(theta0, -math.pi) do
      theta0 = theta0+2*math.pi
   end
   while diff.lt(math.pi, theta0) do
      theta0 = theta0-2*math.pi
   end
   local alpha0 = alpha0
   while diff.lt(alpha0, -math.pi) do
      alpha0 = alpha0+2*math.pi
   end
   while diff.lt(math.pi, alpha0) do
      alpha0 = alpha0-2*math.pi
   end
   local function f(Z)
      local R = R(Z,Psi)
      return theta(R,Z)-theta0
   end
   local Zval = root.ridders(f, -Ztop, Ztop, stop(1e-7))
   local Rval = R(Zval, Psi)
   local phival = alpha0 - alpha(Rval, Zval, 0)
   while diff.lt(phival, -math.pi) do
      phival = phival+2*math.pi
   end
   while diff.lt(math.pi, phival) do
      phival = phival-2*math.pi
   end
   return Rval, Zval, phival
end

-- compute bmag = grad(alpha) X grad(psi)
local function bmagPsiAlpha(R, Z, phi)
   local phi = phi or 0 -- bmag is axisymmetric, so there should be no phi dependence
   local grad_Psi_R, grad_Psi_Z, grad_Psi_phi = grad_Psi(R,Z)
   local grad_alpha_R, grad_alpha_Z, grad_alpha_phi = grad_alpha(R,Z,phi)

   local B_R = grad_alpha_Z*grad_Psi_phi - grad_alpha_phi*grad_Psi_Z
   local B_Z = grad_alpha_phi*grad_Psi_R - grad_alpha_R*grad_Psi_phi
   local B_phi = grad_alpha_R*grad_Psi_Z - grad_alpha_Z*grad_Psi_R
   return math.sqrt(B_R^2 + B_Z^2 + B_phi^2), B_R, B_Z, B_phi
end

-- compute bmag = I(Psi)*grad(phi) + grad(psi) X grad(phi)
local function bmagRZ(R, Z, phi)
   local phi = phi or 0 -- bmag is axisymmetric, so there should be no phi dependence
   local grad_Psi_R, grad_Psi_Z, _ = grad_Psi(R,Z)
   local B_phi = Bphi(R)
   local B_R = grad_Psi_Z/R
   local B_Z = -grad_Psi_R/R
   return math.sqrt(B_R^2 + B_Z^2 + B_phi^2), B_R, B_Z, B_phi
end




xLower = 0.1
xUpper = 1.0
Lx = xUpper - xLower
Ly = 50*rho_s 
Lz = 2*math.pi
print("Ly  = ", Ly)

local r0 = R(0, 0.5) - R0
q0 = 2
--local ntoroidal = 2*math.pi*r0/q0/Ly
--local ntoroidal = 2*math.pi*r0/q0/Ly*96/3
local Ny = 96
local ntoroidal = 2*math.pi*r0/q0/Ly *96/Ny

--Ly = Ly*q0/r0
--print("Ly  = ", Ly)

-- Source parameters.
P_SOL        = 1.0e6*200/3 -- [W], total SOL power, from experimental heating power
P_src        = P_SOL/ntoroidal -- [W], fraction of total SOL power into flux tube
--xSource      = R(0,1.25*PsiSep) - Rsep -- [m], source start coordinate
xSource = 0.35
lambdaSource = 0.03*Lx -- [m], characteristic length scale of density and temperature

-- Source profiles.
sourceDensity = function (t, xn)
   local x, y, z = xn[1], xn[2], xn[3]
   local sourceFloor = 1e-10
   if x < xSource then
      sourceFloor = 1e-2 -- higher floor to left of source peak
   end
   if math.abs(z) < Lz/4 then
      return math.max(math.exp(-(x-xSource)^2/(2*lambdaSource)^2), sourceFloor)
   else
      return 1e-40
   end
end
sourceTemperature = function (t, xn)
   local x, y, z = xn[1], xn[2], xn[3]
   if x < xSource + 3*lambdaSource then
      return 80*eV
   else
      return 30*eV
   end
end

-- Initialize a random seed for initial conditions
-- will be used for both ions and electrons.
randomseed = 100000*Mpi.Comm_rank(Mpi.COMM_WORLD)+63--os.time()

local finalTime = 1.e-3/10 * 8
--local numFrames = 5000/10
local numFrames = 100 * 8

plasmaApp = Plasma.App {
   logToFile = true,

   tEnd        = finalTime,                        -- End time.
   nFrame      = numFrames,                        -- Number of output frames.
   --lower       = {xLower, -Ly/2, -math.pi +1e-2},        -- Configuration space lower left.
   --upper       = {xUpper,  Ly/2,  math.pi-1e-2},  -- Configuration space upper right.

   --lower       = {xLower, -Ly/2/96*Ny, -math.pi +1e-2},        -- Configuration space lower left.
   --upper       = {xUpper,  Ly/2/96*Ny,  math.pi-1e-2},  -- Configuration space upper right.

   --lower       = {xLower, -Ly/2/96*Ny, -2.5},        -- Configuration space lower left.
   --upper       = {xUpper,  Ly/2/96*Ny,  2.5},  -- Configuration space upper right.
   
   lower       = {xLower, -Ly/2*q0/r0 /96*Ny, -math.pi +1e-2},        -- Configuration space lower left.
   upper       = {xUpper, Ly/2*q0/r0  /96*Ny,  math.pi-1e-2},  -- Configuration space upper right.

   --lower       = {xLower, -0.003, -3.14},        -- Configuration space lower left.
   --upper       = {xUpper, 0.003,  3.14},  -- Configuration space upper right.

   cells       = {16, Ny, 16},                     -- Configuration space cells.
   mapc2p = function(xc)
      -- field-aligned coordinates (x,y,z)
      local x, y, z = xc[1], xc[2], xc[3]
      -- x = Rmid - Rsep is distance from separatrix at midplane (Z=0)
      --local Rmid = x + Rsep
      local Psi = x--Psi(Rmid,0)
      local theta = z
      local alpha = y
      -- cylindrical coordinates (R,phi,Z)
      local R, Z, phi = RZphi(Psi, theta, alpha)
      -- cartesian coordinates (X,Y,Z)
      local X = R*math.cos(phi)
      local Y = R*math.sin(phi)
      return X, Y, Z
   end,
   basis       = "serendipity",            -- One of "serendipity" or "maximal-order".
   polyOrder   = 1,                        -- Polynomial order.
   timeStepper = "rk3s4",                    -- One of "rk2" or "rk3".
   cflFrac     = 0.5,
   decompCuts = {1, 1, 1},

   hasRZ = true,
   rzBasis = "serendipity",
   rzPolyOrder = 2,

   rzlower       = {0.01, -6.0 }, -- Configuration space lower left.
   rzupper       = {6.0, 6.0},    -- Configuration space upper right.
   rzcells       = {64, 128},                 -- Configuration space cells.

   -- Boundary conditions for configuration space.
   periodicDirs = {2},     -- Periodic in y only.

   restartFrameEvery = .05,
   groupDiagnostics = true,
   --parallelizeSpecies=true,

   -- Gyrokinetic electrons.
   electron = Plasma.Species {
      charge = qe,
      mass  = me,
      lower = {-4*vte, 0},
      upper = { 4*vte, 12*me*vte^2/(2*B0)},
      cells = {10, 5},
      -- Initial conditions.
      init = Plasma.MaxwellianProjection {
         density = function (t, xn)
            local x, y, z, vpar, mu = xn[1], xn[2], xn[3], xn[4], xn[5]
            local Ls              = Lz/4
            local floor = 0.1
            local effectiveSource = math.max(sourceDensity(t,{x,y,0}),floor)
            local c_ss            = math.sqrt(5/3*sourceTemperature(t,{x,y,0})/mi)
            local nPeak           = 4*math.sqrt(5)/3/c_ss*Ls*effectiveSource/2
            local perturb         = 0 
            if math.abs(z) <= Ls then
               return nPeak*(1+math.sqrt(1-(z/Ls)^2))/2*(1+perturb)
            else
               return nPeak/2*(1+perturb)
            end
         end,
         temperature = function (t, xn)
            local x = xn[1]
            if (x < xSource + 3*lambdaSource) then 
               return 50*eV
            else 
               return 20*eV
            end
         end,
         scaleWithSourcePower = true,
      },
      coll = Plasma.LBOCollisions { 
         collideWith = {'electron', 'ion'},
         frequencies = {nuElc, nuElcIon},
      },
      source = Plasma.Source {
         kind        = "Maxwellian",       power       = P_src/2,
         density     = sourceDensity,      diagnostics = {"M0", "Temp", "intKE"},
         temperature = sourceTemperature,
      },
      polarizationDensityFactor = n0,
      evolve = true, -- Evolve species?
      ---nDistFuncFrame = 10,
      --applyPositivity = true,
      diagnostics = {"M0", "Upar", "Temp", "Beta", "intM0", "intM1", "intEnergy" }, 
      randomseed = randomseed,
      bcx = {Plasma.AbsorbBC{}, Plasma.AbsorbBC{}},
      bcz = {Plasma.SheathBC{diagnostics={"M0"}},
             Plasma.SheathBC{diagnostics={"M0"}}},
   },

   -- Gyrokinetic ions
   ion = Plasma.Species {
      charge = qi,
      mass   = mi,
      -- Velocity space grid.
      lower = {-4*vti, 0},
      upper = { 4*vti, 12*mi*vti^2/(2*B0)},
      cells = {10, 5},
      -- Initial conditions.
      init = Plasma.MaxwellianProjection {
         density = function (t, xn)
            local x, y, z         = xn[1], xn[2], xn[3]
            local Ls              = Lz/4
            local floor = 0.1
            local effectiveSource = math.max(sourceDensity(t,{x,y,0}),floor)
            local c_ss            = math.sqrt(5/3*sourceTemperature(t,{x,y,0})/mi)
            local nPeak           = 4*math.sqrt(5)/3/c_ss*Ls*effectiveSource/2
            local perturb         = 0
            if math.abs(z) <= Ls then
               return nPeak*(1+math.sqrt(1-(z/Ls)^2))/2*(1+perturb)
            else
               return nPeak/2*(1+perturb)
            end
         end,
         temperature = function (t, xn)
            local x = xn[1]
            if x < xSource + 3*lambdaSource then 
               return 50*eV
            else 
               return 20*eV
            end
         end,
         scaleWithSourcePower = true,
      },
      coll = Plasma.LBOCollisions { 
         collideWith = {'ion', 'electron'},
         frequencies = {nuIon, nuIonElc},
      },
      source = Plasma.Source {
         density     = sourceDensity,      power       = P_src/2,
         temperature = sourceTemperature,  isSource    = true,
         diagnostics = {"M0", "Temp", "intKE"},
      },
      polarizationDensityFactor = n0,
      evolve = true, -- Evolve species?
      --nDistFuncFrame = 10,
      --applyPositivity = true,
      diagnostics = {"M0", "Upar", "Temp", "intM0", "intM1", "intEnergy"}, 
      randomseed = randomseed,
      bcx = {Plasma.AbsorbBC{}, Plasma.AbsorbBC{}},
      bcz = {Plasma.SheathBC{diagnostics={"M0"}},
             Plasma.SheathBC{diagnostics={"M0"}}},
   },

   -- Field solver.
   field = Plasma.Field {
      bcLowerPhi  = {{T = "D", V = 0.0}, {T = "P"}}, 
      bcUpperPhi  = {{T = "D", V = 0.0}, {T = "P"}},
      evolve = true, -- Evolve fields?
      isElectromagnetic = false,
   },

   -- Magnetic geometry.
   funcField = Plasma.Geometry {
      -- Read pre-generated geo file. 
      -- To compute geo from scratch, comment out this line (it's expensive).
      --fromFile = "allGeo.in",

      -- Background magnetic field.
      bmag = function (t, xn)
         local x, z = xn[1], xn[3]
         local R, Z = RZ(x, z)
         local bmag, _, _, _ = bmagRZ(R,Z)
         return bmag
      end,

      -- Geometry is not time-dependent.
      evolve = false,
      hasRZ=true,

      psifunc = function(t, xn) 
         local R = xn[1]
         local Z = xn[2]
         return Psi(R,Z)
      end,

      psibyrfunc = function(t, xn) 
         local R = xn[1]
         local Z = xn[2]
         return Psi(R,Z)/R
      end,

      psibyr2func = function(t, xn) 
         local R = xn[1]
         local Z = xn[2]
         return Psi(R,Z)/R/R
      end,

      bphifunc = function(t, xn) 
         local R = xn[1]
         return Bphi(R)
      end,

      B0 = B0,
      R0 = R0,

   },
}
-- Run application.
plasmaApp:run()
