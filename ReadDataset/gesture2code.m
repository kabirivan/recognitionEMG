function code = gesture2code(gesture)

    switch gesture
       
        case 'noGesture'
            code = 1;
            
        case 'fist'
            code = 2; 
            
        case 'waveIn'
            code = 3;  
            
        case 'waveOut'
            code = 4;    
         
        case 'open'
            code = 5;
            
        case 'pinch'
            code = 6;
      
            
    end
end
