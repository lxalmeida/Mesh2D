#include <topology_output.h>

void Topology_output::print_info(Topology &topology){
    ostringstream filename(ostringstream::out);
    struct dims dims = topology.get_dims();
    struct coords coords = topology.get_coords();
    
    filename << "output_topology." << topology.get_rank() << ".txt";

    ofstream fout(filename.str().c_str());
    
    fout << "ndims = " << topology.get_ndims() << endl;
    switch(topology.get_ndims()) {
        case 1:
            fout << "dims  = " << dims.x << endl;
            fout << "coords = " << coords.x << endl;
            break;
        case 2:
            fout <<  "dims  = " << dims.x << " x " << dims.y << endl;
            fout <<  "coords = (" << coords.x << ", " << coords.y << ")" << endl;
            break;
        case 3:
            fout <<  "dims  = " << dims.x << " x " << dims.y << " x " << dims.z << endl;
            fout <<  "coords = (" << coords.x << ", " << coords.y << ", " << coords.z << ")" << endl;
            break;
    }
    fout <<  "rank   = " << topology.get_rank() << endl;
    fout <<  "nprocs = " << topology.get_nprocs() << endl;

    fout << "Neighbors: " << endl;
    neighbors_iterator neighbor = topology.first_neighbor();
    
    while(neighbor != topology.last_neighbor()){
        switch((*neighbor).direction){
            case RIGHT:
                fout << " right_rank: " << (*neighbor).rank << endl;
                break;
            case LEFT:
                fout << " left_rank: " << (*neighbor).rank << endl;
                break;
            case TOP:
                fout << " top_rank: " << (*neighbor).rank << endl;
                break;
            case BOTTOM:
                fout << " bottom_rank: " << (*neighbor).rank << endl;
                break;
            case TOP_RIGHT:
                fout << " top_right_rank: " << (*neighbor).rank << endl;
                break;
            case TOP_LEFT:
                fout << " top_left_rank: " << (*neighbor).rank << endl;
                break;
            case BOTTOM_RIGHT:
                fout << " bottom_right_rank: " << (*neighbor).rank << endl;
                break;
            case BOTTOM_LEFT:
                fout << " bottom_left_rank: " << (*neighbor).rank << endl;
                break;
        }
        
        neighbor++;
    }
        
    fout.close();
    
    return;
}

