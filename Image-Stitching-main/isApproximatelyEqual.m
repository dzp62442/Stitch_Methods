function out = isApproximatelyEqual(a, b, threshold)
    if(abs(a-b)<=threshold)
        out = true;
    else
        out = false;
    end
end