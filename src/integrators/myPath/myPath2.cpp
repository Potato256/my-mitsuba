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
    enum SamplingStrategy {
        PathBSDF = 0,
        PathNEE,
        PathMIS
    };

private:    
    int m_maxDepthEye;
    Float m_rrEye;

    ref<Bitmap> m_bitmap;
    int m_blockSize;
    bool m_running;

    std::vector<Point2i> blockOfs;

public:
    /// Initialize the integrator with the specified properties
    myPath2Integrator(const Properties &props) : Integrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_blockSize = props.getInteger("blockSize", 100);

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
        oss<<"\n--------- myPath2 Info Print ----------\n";
        oss<<"maxDepthEye: "<<m_maxDepthEye<<endl;
        oss<<"rrEye: "<<m_rrEye<<endl;
        oss<<"blockSize: "<<m_blockSize<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);

        blockOfs.clear();
        const Film* film = scene->getSensor()->getFilm();  
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();
        int w = cropSize.x / m_blockSize + 1;
        int h = cropSize.y / m_blockSize + 1;
        for (int i = 0; i < w; ++i){
            for (int j = 0; j < h; ++j){
                blockOfs.push_back(
                    Point2i(cropOffset.x+i*m_blockSize, cropOffset.y+j*m_blockSize));
            }
        }

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

        // std::cout<<cropOffset.toString()<<std::endl;

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

        for (int i = 0; i < sampleCount; ++i) {
            std::cout << "Frame: " << i << std::endl;
            if (!m_running) 
                break;
            /* Trace eye subpath*/
            int blockCnt = (int) blockOfs.size();
            #if defined(MTS_OPENMP)
                #pragma omp parallel for schedule(dynamic)
            #endif
            for (int block = 0; block < blockCnt; ++block) {
                Point2i& bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;

                for (int yofs=0; yofs<m_blockSize; ++yofs) {
                    for (int xofs=0; xofs<m_blockSize; ++xofs) {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        
                        if (needsApertureSample)
                            apertureSample = sampler->next2D();
                        samplePos = sampler->next2D();
                        
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;
    
                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);   
                        
                        int ofs = yRealOfs*cropSize.x+xRealOfs;
                        Spectrum L = Li(eyeRay, scene, sampler);
                        float i_ = (float) i;
                        target[ofs] = (target[ofs]*i_ + L)/(i_+1.0f);
                    }
                }

            }
            film->setBitmap(m_bitmap);       
            queue->signalRefresh(job); 
        }

        printInfos();
        
        return true;
    }

    inline Float mis(Float p1, Float p2) const {
        return p1 / (p1 + p2);
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, Scene* scene, Sampler* sampler) const {
        /* Some aliases and local variables */
        Intersection its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        scene->rayIntersect(ray, its);
        ray.mint = Epsilon;

        if (its.isValid() && its.isEmitter())
            return its.Le(-ray.d);

        int depth = 1;
        while(depth <= m_maxDepthEye) {
            if (!its.isValid())
                break;
            
            const BSDF *bsdf = its.getBSDF(ray);
            DirectSamplingRecord dRec(its);

            if (bsdf->getType() & BSDF::ESmooth) {
                Spectrum value = scene->sampleEmitterDirect(dRec, sampler->next2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero()) {
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;
                        Float misW = mis(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * misW;
                    }
                }
            }

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sampler->next2D());
            if (bsdfWeight.isZero())
                break;
            
            const Vector wo = its.toWorld(bRec.wo);
            
            throughput *= bsdfWeight ;
            eta *= bRec.eta;

            ray = Ray(its.p, wo, ray.time);

            Spectrum value;
            if (scene->rayIntersect(ray, its)) {
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                        scene->pdfEmitterDirect(dRec) : 0;
                    Float misW = mis(bsdfPdf, lumPdf);
                    Li += throughput * value * misW;
                    return Li;
                }
            }

            Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
            if (sampler->next1D() >= q)
                break;
            throughput /= q;
            ++depth;
        }
        return Li;
    }

};

MTS_IMPLEMENT_CLASS_S(myPath2Integrator, false, Integrator)
MTS_EXPORT_PLUGIN(myPath2Integrator, "myPath2");
MTS_NAMESPACE_END