function [rxtshfc, rxtssfc, csi_store, sncsis, txtshfc, packets, correz, csi_store_aoa, rxtshfc_aoa, rxtssfc_aoa] = load80MHZstudio(FILE, BW, nodeid)
 
%% csireader.m
%
% read and plot CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
%

%% configuration
CHIP = '4366c0';          % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
% BW = 80;                % bandwidth
% FILE = './trace10_80M.pcap';

HOFFSET = 16;           % header offset
NFFT = BW*3.2;          % fft size
p = readpcap();
p.open(FILE);
n = length(p.all());
p.from_start();
k = 0;

% use this to count how many bits ~= 0 in a nibble
% access it with nibble + 1
bits_counter = [0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4];

% status
prevfwcnt = -1;
mask_to_process = -1;
processedmask = mask_to_process;
processedpacket = 0;
chars='|/-\';
output = '';

NICE = 1;
if NICE == 1,
  fprintf(1, '.');
end;

prevfwmask = -1;

csi_store = {};
packets = 1;
tshfc = [];
tssfc = [];
metadatas = [];
sncsis = [];

txtsdata = [];
framecsi = {};

id = hex2dec(nodeid([3 4 1 2]));

capra = [];

while (k < n)
    k = k + 1;
    if NICE == 1,
      for jj = 1:length(output) + 1, fprintf(1, '\b'); end;
    end;
    charjj = mod(k, length(chars)) + 1;
    output = sprintf(' %.0f/100', k / n * 100);
    if NICE == 1,
      fprintf(1, '%c%s', chars(charjj), output);
    end;
    f = p.next();
    if isempty(f)
        disp('no more frames');
        return;
    end

    % we can have wireless packet or the udp stuff with csi
    % wireless packet starts with prism header, assume is always [4100 0000 9000]
    wirelesstype = uint16([65 0 144])';
    % udp starts with broadcast address [ffff ffff ffff]
    udptype = uint16([65535 65535 65535])';
    
    LEN = length(f.payload);
    LEN = 2 * floor(LEN / 2);
    data = typecast(f.payload(1:LEN), 'uint16');
    
%     data = typecast(f.payload(1:end), 'uint16');

    if isequal(data(1:3), wirelesstype),
        % check this is the right packet 
	% we use mac addr 2 as filter: last two bytes as index of station
        % 0x0090:  8802 0000 ffff ffff ffff 0012 3456 789b  ............4Vx.
        expectedheader = uint16([0 648 0 65535 65535 65535 4608 22068 id])';
        if length(data) >= 80 & isequal(data(72:80), expectedheader),
            % extract ts and sequence number of this frame, (we should match it with the
            % sequence number in the next packet with the csi but it's not important right now)
            % just remember the sequence number and the timestamp so that we have the transmission
            % time stamp for that sequence number
            % 0x00a0:  79e5 7c7f 5016 6016 0000 aaaa 0300 0000  y.|.P.`.........
            sn = data(83);
            txts = typecast(flipud(data(81:82)), 'uint32');
	    capra = [capra ; txts];
            txtsdata = [txtsdata; [uint32(sn) txts]];
        end;
        continue;
    end;

    % from this point we are processing the udp containing the csidata
    if ~ isequal(data(1:3), udptype),
        hola
        continue;
    end;

    % for quick reference:
    %							byte offset in packet
    % payload(1)  first  w32: data(1)  and data(2)	0x0000
    % payload(2)  second w32: data(3)  and data(4)	0x0004
    % ...
    % payload(11) ..........: data(21) and data(22)	0x0028
    % payload(12) ..........: data(23) and data(24)	0x002c
    % payload(13) ..........: data(25) and data(26)	0x0030
    % payload(14) ..........: data(27) and data(28)	0x0034
    % payload(15) ..........: data(29) and data(30)	0x0038
    %
    % packet structure:
    % 0x0022: udp port
    % 0x0024: udp port
    % 0x0026: udp len
    % 0x0028: udp checksum
    % 0x002a: 1111 (magic)
    % 0x002c: configuration of CSI capture (0f 01 in wireshark means 4 cores, 1 nss)
    % 0x002e: mac address of transmitter
    % 0x0034: seq counter of wireless frame
    % 0x0036: csi configuration
    % 0x0038: chanspec
    % 0x003a: firmware counter (increase with frame, not CSI chunks)
    %
    % to recap:
    % payload(10) contains 0x002c and 0x002d
    % payload(14) contains 0x0034 and 0x0036
    % payload(15) contains 0x0038 and 0x003a

    if f.header.orig_len-(HOFFSET-1)*4 < NFFT*4
        disp('skipped frame with incorrect size');
        return;
    end
%     data(26)
    % check if the id of the node is that expected
    if data(26) ~= data(26),
      continue;
    end;

    % extract configuration for this packet, cannot use it now, start from next one (see below)
    magic = data(22);
    csiconfig = data(23);
    
    % extract sequence number... if there are multiple csi for the same packet, it will be the same
    % and we will put into sncsis only once
    sncsi = data(27);

    % extract number of packet processed by the firmware and rxcore for this packet
    fwcnt = double(bitand(f.payload(15), 65535 * 65536) / 65536);
    fwmask = bitand(f.payload(14), 255 * 65536) / 65536;
    current_core_index = double(bitand(fwmask, 12) / 4);
    current_nss_index = double(bitand(fwmask, 3));
%     fwcnt
    % if fwcnt wrapped around, then this is like when we have a new packet
    realprevfwcnt = prevfwcnt;
    if fwcnt < prevfwcnt,
      disp 'fwcnt wrapped around...';
      prevfwcnt = fwcnt - 1;
    end;

    % if this data is for a new packet (like at beginning), extract
    % configuration and compute how many data we should extract
    % report if for the previous packet we did not received all rxcore data
    if fwcnt > prevfwcnt,
      if processedmask < mask_to_process,
        disp(sprintf('\nmissing data for packet %d\n', realprevfwcnt));
        for jj = 1:length(output), fwrite(1, 10); end;
        % drawnow('update');
      end;
      processedmask = 0;
      prevfwcnt = fwcnt;
      prevfwmask = -1;
      core_config = double(bitand(csiconfig, 255));
      nss_config = double(bitand(csiconfig, 65280) / 256);
      core_number = bits_counter(core_config + 1);
      nss_number = bits_counter(nss_config + 1);
      mask_to_process = core_number * nss_number;
      framecsi = {};
      framecsi.mask_to_process = mask_to_process;
      framecsi.nss_config = nss_config;
      framecsi.core_config = core_config;
      metadata = [];
    end;

    if NICE == 0,
      disp(sprintf('%d %d %d %d %d %d %d %d', core_config, nss_config, core_number, nss_number, mask_to_process, current_core_index, current_nss_index, processedmask));
    end;

    processedmask = processedmask + 1;

    % this should not happen
    if mask_to_process == -1,
      disp(sprintf('mask_to_process never initalized'));
      keyboard;
    end;

    % this should not happen
    if processedmask > mask_to_process,
      disp(sprintf('More than %d masks for this packet, terminating...', mask_toprocess));
      keyboard;
    end;

    % this should not happen
    if fwmask < prevfwmask,
      disp(sprintf('\nfwmask goes backward, skipping CSI %d\n', fwcnt));
      for jj = 1:length(output), fwrite(1, 10); end;
      % drawnow('update');
      processedmask = 0;
      prevfwcnt = fwcnt;
      prevfwmask = -1;

      core_config = double(bitand(csiconfig, 255));
      nss_config = double(bitand(csiconfig, 65280) / 256);
      core_number = bits_counter(core_config + 1);
      nss_number = bits_counter(nss_config + 1);
      mask_to_process = core_number * nss_number;
      framecsi = {};
      framecsi.mask_to_process = mask_to_process;
      framecsi.nss_config = nss_config;
      framecsi.core_config = core_config;
      metadata = [];

    end;

    prevfwmask = fwmask;

    % extract data for this CSI
    
%     try
        H = f.payload(HOFFSET:HOFFSET+NFFT-1);
%     catch
%         break;
%     end
        
    Hout = unpack_float(int32(1), int32(NFFT), H);
    Hout = reshape(Hout,2,[]).';
    cmplx = double(Hout(1:NFFT,1))+1j*double(Hout(1:NFFT,2));
    if BW == 20,
      cmplx = cmplx([33:64 1:32]);
    elseif BW == 80,
      cmplx = cmplx([129:256 1:128]);
    else
      disp 'BAND??';
      keyboard
    end;
    framecsi.core{current_core_index + 1}.nss{current_nss_index + 1}.data = cmplx;
    data = typecast(f.payload(HOFFSET+NFFT:end), 'uint16');
    metadata = [metadata data'];

    if processedmask == mask_to_process,
      csi_store{packets} = framecsi;
      sncsis = [sncsis; sncsi];
      metadatas{packets} = metadata;
      tslo = uint32(metadata(26));
      tshi = uint32(metadata(27));
      ts2lo = uint32(metadata(18));
      ts2hi = uint32(metadata(19));
      tshfc = [tshfc; tshi * 65536 + tslo];
      tssfc = [tssfc; ts2hi * 65536 + ts2lo];
      packets = packets + 1;

    end;

end

packets = packets - 1;

% unwrap tshfc
tshfc_orig = tshfc;

if (isempty(tshfc))
    rxtshfc = [];
    rxtssfc = [];
    csi_store = [];
    sncsis = [];
    txtshfc = [];
    packets = [];
    correz = [];
    csi_store_aoa = [];
    rxtshfc_aoa = [];
    rxtssfc_aoa = [];
    
    return;
else
tshfc = uint32(2) ^ 32 - tshfc;
wrapidx = find(diff(double(tshfc)) < 0);
tshfc = uint64(tshfc);
for kk = 1:length(wrapidx),
  adder = uint64(zeros(size(tshfc)));
  adder(wrapidx(kk) + 1:end) = uint64(2) ^ 32;
  tshfc = tshfc + adder;
end;

% unwrap txtsdata
% txtsdata_orig = txtsdata;
% txtsdata(:, 2) = uint32(2) ^ 32 - txtsdata(:, 2);
% wrapidx = find(diff(double(txtsdata(:, 2))) < 0);
% txtsdata = uint64(txtsdata);
% for kk = 1:length(wrapidx),
%   adder = uint64(zeros(size(txtsdata(:, 2))));
%   adder(wrapidx(kk) + 1:end) = uint64(2) ^ 32;
%   txtsdata(:, 2) = txtsdata(:, 2) + adder;
% end;

% now filter out frames with more than 1 CSI mask
masknos = zeros([length(csi_store) 1]);
for kk = 1:length(csi_store),
  masknos(kk) = csi_store{kk}.mask_to_process;
end;

dropper = find(masknos > 1);
csi_store_aoa = csi_store(dropper);
rxtshfc_aoa = tshfc(dropper);
rxtssfc_aoa = tssfc(dropper);
keepidxs = 1:length(csi_store);
keepidxs(dropper) = [];
tshfc = tshfc(keepidxs);
tssfc = tssfc(keepidxs);
csi_store = csi_store(keepidxs);
sncsis = sncsis(keepidxs);
metadatas = metadatas(keepidxs);

% remove wrap around in the sequence counters
sncsis_orig = sncsis;
sncsis = uint32(sncsis);
wrapidx = find(diff(double(sncsis)) < 0);
for kk = 1:length(wrapidx),
  adder = uint32(zeros(size(sncsis)));
  adder(wrapidx(kk) + 1:end) = 65536;
  sncsis = sncsis + adder;
end;
% txtsdata_orig = txtsdata;
% if ~isempty(txtsdata),
%   wrapidx = find(diff(double(txtsdata(:, 1))) < 0);
%   for kk = 1:length(wrapidx),
%     adder = uint64(zeros(size(txtsdata(:, 1))));
%     adder(wrapidx(kk) + 1:end) = 65536;
%     txtsdata(:, 1) = txtsdata(:, 1) + adder;
%   end;
% end;

% now we have two datasets
% 1) tshfc, tssfc, csi_store, sncsis
% 2) txtsdata
% we should keep only packets for which we have both tx and rx timestamps
rxtshfc = tshfc;
rxtssfc = tssfc;
% if ~isempty(tssfc),
%   [cmn, ia, ib] = intersect(uint64(sncsis), txtsdata(:, 1));
%   rxtshfc = tshfc(ia);
%   rxtssfc = tssfc(ia);
%   csi_store = csi_store(ia);
%   sncsis = sncsis(ia);
%   metadatas = metadatas(ia);
%   txtshfc = txtsdata(ib, 2);
%   packets  = length(rxtshfc);
% else
%   rxtshfc = [];
%   rxtssfc = [];
%   csi_store = [];
%   sncsis = [];
%   metadatas = [];
%   txtshfc = [];
%   packets = 0;
% end;

% load(sprintf('reference80true.mat', BW)); 
correz = [];
% for kk = 1:packets,
%   % must exist core{0}.nss{0} for this to work
%   csi = transpose(csi_store{kk}.core{1}.nss{1}.data);
% 
%   diffs = diff(unwrap(angle(csi ./ reference)));
% 
%   if BW == 20,
%     left = diffs(7:32);
%     rigt = diffs(34:59);
%   elseif BW == 80,
%     left = diffs(7:127);
%     rigt = diffs(131:251);
%   end;
%   correz = [correz; median(left(2:end - 1)) median(rigt(2:end - 1))];
% end;
% correz = correz / 2 / pi / 312.5e3;

FREQ = 2^26 / 348436 * 1e6;
if ~isempty(correz),
  rxtshfc = double(rxtshfc) / FREQ;%; - correz(:, 1);
end;
% txtshfc = double(txtshfc) / FREQ;
rxtssfc = double(rxtssfc);

rxtshfc_aoa = double(rxtshfc_aoa) / FREQ;
rxtssfc_aoa = double(rxtssfc_aoa);
txtshfc = [];
end
% txtimes = double(txtsdata(:, 2)) / FREQ;
% rxtimes = double(tshfc) / FREQ;
% rxtimes = rxtimes - correz(:, 1);
