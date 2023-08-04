#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

class myPath2Integrator : public Integrator {
public:
    MTS_DECLARE_CLASS()

private:    
    int m_maxDepthEye;
    int m_maxDepthLight;
    Float m_rrEye;
    Float m_rrLight;

    ref<Bitmap> m_bitmap;
    bool m_running;

public:
    /// Initialize the integrator with the specified properties
    myPath2Integrator(const Properties &props) : Integrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_maxDepthLight = props.getInteger("maxDepthLight", 5);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_rrLight = props.getFloat("rrLight", 0.6);

        m_running = true;
    }
    
    // Unserialize from a binary data stream
    myPath2Integrator(Stream *stream, InstanceManager *manager)
        : Integrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
    }

    void printInfos(){
        std::ostringstream oss;
        oss<<"\n---------- LVCBPT Info Print -----------\n";
        oss<<"maxDepthEye: "<<m_maxDepthEye<<endl;
        oss<<"maxDepthLight: "<<m_maxDepthLight<<endl;
        oss<<"rrEye: "<<m_rrEye<<endl;
        oss<<"rrLight: "<<m_rrLight<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        printInfos();

        return true;
    }

    void cancel(){
        m_running = false;
    }
    
    bool render(Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int sensorResID, int unused) {

        ref<Scheduler> sched = Scheduler::getInstance();
        int blockSize = scene->getBlockSize();
        
        ref<Sampler> sampler = scene->getSampler();        
        size_t sampleCount = sampler->getSampleCount();

        ref<Sensor> sensor = scene->getSensor();
        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        ref<Film> film = sensor->getFilm();
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();

        size_t nCores = sched->getCoreCount();
                
        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
            film->getCropSize().x, film->getCropSize().y,
            nCores, nCores == 1 ? "core" : "cores");
            
        /* Create a sampler instance for every core */
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);

        /* Allocate memory */
        m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        m_bitmap->clear();

        int* LVCSize = new int[nCores];

        Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();

        for (int i =0; i < sampleCount; ++i) {

            if (!m_running) 
                break;
            
            /* Trace LVC */
            m_LVCVertexSize = 0;
            memset(LVCSize, 0, sizeof(int)*nCores);
            #define MTS_OPENMP
            #if defined(MTS_OPENMP)
                #pragma omp parallel
                {   
                    int tid = mts_omp_get_thread_num();
                    Sampler *sampler = static_cast<Sampler *>(samplers[tid]);
                    int pSize = m_LVCPathSize / nCores;
                    int vSize = pSize * m_maxDepthLight * tid;
                    for (int i = 0; i < pSize; ++i)
                        traceLightSubpath(sampler, scene, 
                            tmp_LVC + vSize , LVCSize[tid]);
                    #pragma omp critical
                    {
                        int start = m_LVCVertexSize;
                        m_LVCVertexSize += LVCSize[tid];
                        memcpy(m_LVC + start, tmp_LVC + vSize, 
                            sizeof(BDPTVertex) * LVCSize[tid]); 
                    }
                }
            #else
                for (int i = 0; i < m_LVCPathSize; ++i)
                    traceLightSubpath(sampler, scene, m_LVC, m_LVCVertexSize);
            #endif

            /* Trace eye subpath*/
            for (int yofs=0; yofs<cropSize.y; ++yofs) {
                for (int xofs=0; xofs<cropSize.x; ++xofs) {
                    
                    Spectrum Li(0.0f);

                    // if (xofsInt + xofs >= cropSize.x)
                    //     continue;
                    Point2 apertureSample, samplePos;
                    Float timeSample = 0.0f;
                    
                    if (needsApertureSample)
                        apertureSample = sampler->next2D();
                    if (needsTimeSample)
                        timeSample = sampler->next1D();
                    samplePos = sampler->next2D();
                    
                    samplePos.y += cropOffset.y + yofs;
                    samplePos.x += cropOffset.x + xofs;

                    RayDifferential eyeRay;
                    
                    Spectrum sampleValue = sensor->sampleRay(
                        eyeRay, samplePos, apertureSample, timeSample);                    


                    /* Trace first eye vertex */
                    RayDifferential eyeRay(r);
                    rRec.rayIntersect(eyeRay);
                    eyeRay.mint = Epsilon;

                    /* Hit nothing */
                    if (!rRec.its.isValid())
                        return Li;        
                    /* Hit emitter */
                    if ( rRec.its.isEmitter())
                        return rRec.its.Le(-eyeRay.d);

                    std::vector<BDPTVertex*> eyePath;
                    /* Trace eye subpath */
                    traceEyeSubpath(eyeRay, rRec.its, rRec.sampler, rRec.scene, eyePath, Li);

                    /* Connect eye subpath and light subpath */
                    connectSubpaths(eyePath, lightPath, rRec.scene, Li);

                    /* Clean up */
                    for (auto i : eyePath) delete i;

                    target[yofs*cropSize.x+xofs] = Spectrum(1.0f);
                }
            }
            film->setBitmap(m_bitmap);       
            queue->signalRefresh(job); 
        }

        printInfos();
        
        return true;
    }

};

MTS_IMPLEMENT_CLASS_S(myPath2Integrator, false, Integrator)
MTS_EXPORT_PLUGIN(myPath2Integrator, "myPath2");
MTS_NAMESPACE_END