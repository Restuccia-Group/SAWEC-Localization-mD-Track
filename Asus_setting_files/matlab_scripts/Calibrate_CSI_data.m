close all
clc
clear

 grid_toa = [-122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, ...
-109, -108, -107, -106, -105, -104, -102, -101, -100, -99, -98, -97, -96, -95, ...
-94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, ...
-78, -77, -76, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, ...
-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, ...
-45, -44, -43, -42, -41, -40, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, ...
-28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, ...
-12, -10, -9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, ...
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, ...
40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, ...
62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, ...
85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, ...
106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, ...
122];
     
subfolder_name = "020822";
     
d = dir(strcat("../mat_files/", subfolder_name, "/"));
d_cell = struct2cell(d);
folder = string(d_cell(1,3:end).')% folder = ["breathing1"];
for id_folder = 1:length(folder)

    % load the template
    load(strcat("../mat_files/", subfolder_name, "/", subfolder_name, "_calibration/", "csi_data.mat"))

    % take the first packet
    csi_template = squeeze(csi_data(1,:,:));
    csi_template = csi_template(grid_toa + 256/2 + 1,:);
    % load the csi data to be calibrated
    load(strcat("../mat_files/", subfolder_name, "/", folder(id_folder), "/csi_data.mat"));

%     % calibrate and save
%     [snapshots, K, N] = size(csi_data);

    % check if we have 0s
    index_0s = csi_data == 0;
    index_0s = sum(squeeze(sum(index_0s,2)),2);
    index_0s = index_0s == 256;
    csi_data(index_0s,:,:) = [];
    % calibrate and save
    [snapshots, K, N] = size(csi_data);

    % remove non-active subcarriers
    csi_data = csi_data(:,grid_toa + 256/2 + 1,:);

    % calibrate the data. Apply the de-convolution and set the absolute
    % value to 1
    for rx_id = 1:N
        csi_data(:,:,rx_id) = squeeze(csi_data(:,:,rx_id))./ squeeze(csi_template(:,rx_id)).';
        csi_data(:,:,rx_id) = squeeze(csi_data(:,:,rx_id))./ mean(abs(squeeze(csi_data(:,:,rx_id))),2);
    end

    % put nans to the non-active subcarriers
    csi_data_aux = nan(snapshots,K,N);
    csi_data_aux(:,grid_toa + (K/2) + 1,:) = csi_data;
    csi_data = csi_data_aux;

    % save it
    save(strcat("../mat_files/", subfolder_name, "/", folder(id_folder), ...
        "/csi_data_calibrated"), "csi_data");

end