#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

template <typename T, size_t Dims>
class RingBuffer{
private:
    std::array<T, Dims> ringbuffer_;
    size_t idxhead_ = 0;    // head pointer
    size_t idxtail_ = 0;  //tail pointer
    size_t idxpop_ = idxtail_;
    size_t idxevent_ = 0; // event pointer, invalid initial value
    bool flgevent_ = false;      // event occur and record NumAftEvent_c section signal
    bool flglock_ = false;       // write lock until writing into flash or manual reset

    size_t shiftidx(size_t idx, uint16_t numshift){
        if(numshift > 0)
            return (idx + numshift) % Dims;
        else
            return (idx + numshift + Dims) % Dims;
    }

public:
    size_t NumAftEvent_c = 0;     // record length after event
    struct PopStatus{
        bool flgrstpop = true;            // reset pop after pop empty
        T item_;
        size_t numleft = 0;
    } popstatus;

    RingBuffer(size_t numAftEvent_c) : NumAftEvent_c(numAftEvent_c) {}      // Constructor
    ~RingBuffer() = default;

    // setevent: only record short section after event occurs
    bool setevent(){
        if(flglock_==true || flgevent_==true)
            return false;
        else {
            flgevent_ = true;
            idxevent_ = idxhead_;
            return true;
        }
    }

    // push: write into buffer 
    int64_t push(const T& item){
        if(flglock_ == true)
            return -1;
        else if(flgevent_== true){
            flglock_ = ( idxhead_ == shiftidx(idxevent_, NumAftEvent_c) ) ;
            if(flglock_ == true)
                return -1;
            else ; 
        }else ;         

        ringbuffer_[idxhead_] = item;
        
        idxhead_ = (idxhead_ +1) % Dims;
        if(idxtail_ == idxhead_)        // push forward idxtail
            idxtail_ = (idxtail_+1) % Dims;

        return idxhead_;
    }

    // pop: read buffer from old to new
    const PopStatus& pop(const int8_t& start){
        size_t idxpop = idxpop_;

        if(popstatus.flgrstpop == true){
            idxpop = idxtail_;
            popstatus.flgrstpop = false;
        }else{
            switch (start){
            case 0:             // from oldest
                idxpop = idxtail_;
                break;
            case -1:            // continue
            default:
                idxpop = (idxpop + 1) % Dims;
                break;
            }
        }
        idxpop_ = idxpop;

        popstatus.item_ = ringbuffer_[idxpop_];
        popstatus.numleft = (idxhead_ - idxpop_ + Dims) % Dims;

        if(idxpop_ == idxhead_)           // pop empty, reset
            popstatus.flgrstpop = true;

        return popstatus;
    }

    // reset ringbuffer
    bool reset(){
        idxhead_ = 0;    // head pointer
        idxtail_ = 0;  //tail pointer
        idxevent_ = 0; // event pointer
        idxpop_ = idxtail_;
        flgevent_ = false;
        flglock_ = false; 
        popstatus.flgrstpop = true;

        return true;
    }
};

int main(){

    RingBuffer<double, 150> bffrfault_data(50);

    for(size_t k=0; k != 1000; k++){

        if(k==500)
            bffrfault_data.setevent();

        bffrfault_data.push(k);
    }

    while(true){
        const auto& [flgrstpop, value, numleft] = bffrfault_data.pop(-1);
        if(flgrstpop == true) 
            break;
    }

    bffrfault_data.reset();

    return 0;
}