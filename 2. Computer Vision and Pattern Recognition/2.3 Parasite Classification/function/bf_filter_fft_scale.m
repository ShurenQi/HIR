function y = bf_filter_fft_scale(fft_x,fft_pbf_cell,bfdata)

    y = ifft2(single(fft_pbf_cell).*single(fft_x));
    y = y(size(bfdata,1):end,size(bfdata,2):end,:);

end