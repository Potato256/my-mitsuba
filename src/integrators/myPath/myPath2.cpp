#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <iostream>
#include <fstream>
#include <time.h>

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
    enum MISMode {
        UniformHeuristic = 0,
        BalanceHeuristic,
        PowerHeuristic
    };
private:    
    int m_maxDepthEye;
    Float m_rrEye;

    ref<Bitmap> m_bitmap;
    std::vector<Point2i> blockOfs;
    int m_blockSize;
    bool m_jitterSample;
    bool m_running;
    bool m_drawCurve;   

    std::string m_strategyString;
    SamplingStrategy m_strategy;
    std::string m_MISmodeString;
    MISMode m_MISmode;

    double m_connectNum = 0;
    double m_connectTime = 0;

public:
    /// Initialize the integrator with the specified properties
    myPath2Integrator(const Properties &props) : Integrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_blockSize = props.getInteger("blockSize", 64);
        m_jitterSample = props.getBoolean("jitterSample", true);
        m_running = true;    
        
        m_strategyString = props.getString("strategy", "mis");
        if (m_strategyString == "bsdf")
            m_strategy = PathBSDF;
        else if (m_strategyString == "nee")
            m_strategy = PathNEE;
        else if (m_strategyString == "mis")
            m_strategy = PathMIS;
        else
            Log(EError, "Unknown strategy: %s", m_strategyString.c_str());

        m_MISmodeString = props.getString("MISmode", "balance");
        if (m_MISmodeString == "uniform")
            m_MISmode = UniformHeuristic;
        else if (m_MISmodeString == "balance")
            m_MISmode = BalanceHeuristic;
        else if (m_MISmodeString == "power")
            m_MISmode = PowerHeuristic;
        else
            Log(EError, "Unknown MIS mode: %s", m_MISmodeString.c_str());
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
        oss<<"connect number: "<<m_connectNum<<endl;
        // oss<<"time per connect: "<<m_connectTime/m_connectNum/CLOCKS_PER_SEC*1000*1000<<"us"<<endl;
        oss<<"time per connect: "<<m_connectTime/m_connectNum<<"ns"<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EInfo, oss.str().c_str());
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

        size_t nCores = sched->getCoreCount();
                
        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
            film->getCropSize().x, film->getCropSize().y,
            nCores, nCores == 1 ? "core" : "cores");
            
        /* Create a sampler instance for every core */
        std::vector<Sampler *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        /* Allocate memory */
        m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        m_bitmap->clear();

        double* cTimes = new double[nCores];
        double* cNums = new  double[nCores];
        memset(cTimes, 0, nCores*sizeof(double));
        memset(cNums, 0,  nCores*sizeof(double));

        Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
        std::string convergeCurve = "";
        m_drawCurve = false;
        if (cropSize.x == 1 && cropSize.y == 1)
            m_drawCurve = true;

        for (int i = 0; i < sampleCount; ++i) {
            if (m_drawCurve){
                if (i%1000==0)
                    SLog(EInfo, "Frame: %i", i);
            }
            else
                SLog(EInfo, "Frame: %i", i);
            if (!m_running) 
                break;
            /* Trace eye subpath*/
            int blockCnt = (int) blockOfs.size();
            #if defined(MTS_OPENMP)
                #pragma omp parallel for schedule(dynamic)
            #endif
            for (int block = 0; block < blockCnt; ++block) {
                int tid = mts_omp_get_thread_num();
                Sampler* sampler = samplers[tid];
                Point2i& bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;

                cNums[tid] = 0.0f;
                cTimes[tid] = 0.0f;
                
                for (int yofs=0; yofs<m_blockSize; ++yofs) {
                    for (int xofs=0; xofs<m_blockSize; ++xofs) {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        
                        if (needsApertureSample)
                            apertureSample = sampler->next2D();
                        samplePos = Point2(0.5, 0.5);
                        if (m_jitterSample)
                            samplePos = sampler->next2D();
                        
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;
    
                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);   
                        
                        int ofs = yRealOfs*cropSize.x+xRealOfs;
                        Spectrum L = Li(eyeRay, scene, sampler, &cTimes[tid], &cNums[tid]);
                        float i_ = (float) i;
                        target[ofs] = (target[ofs]*i_ + L)/(i_+1.0f);
                    }
                }
                #pragma omp critical
                {
                    m_connectNum += cNums[tid];
                    m_connectTime += cTimes[tid];
                }
            }
            film->setBitmap(m_bitmap);       
            queue->signalRefresh(job); 
            if (m_drawCurve)
                convergeCurve += target[0].toString() + "\n";
        }
        printInfos();
        if (m_drawCurve){
            std::ofstream fout;
            std::ostringstream save;
            save << "./experiments/results/";
            save << (m_jitterSample ? "jitter/" : "nojitter/");
            switch (m_strategy)
            {
                case PathBSDF: save << "bsdf";                    break;
                case PathNEE:  save << "nee" ;                    break;
                case PathMIS:  save << "mis-" << m_MISmodeString; break;
            }
            float e = round(log10(sampleCount));
            save << "-" << round(sampleCount/pow(10,e));
            save << "e" << round(log10(sampleCount)) << ".txt";
            fout.open(save.str().c_str());
            fout << convergeCurve;
            fout.close();
            SLog(EInfo, "Saving result to %s", save.str().c_str());
        }

        delete[] cTimes;
        delete[] cNums;

        return true;
    }

    inline Float misWeight(Float pdfBSDF, Float pdfDirect, SamplingStrategy strategy) const {
        switch (strategy)
        {
        case PathBSDF:
            switch (m_strategy)
            {
            case PathBSDF: return 1;
            case PathNEE: return 0;
            case PathMIS: return mis(pdfBSDF, pdfDirect, m_MISmode);
            default: return 0;
            }
        case PathNEE:
            switch (m_strategy)
            {
            case PathBSDF: return 0;
            case PathNEE: return 1;
            case PathMIS: return mis(pdfDirect, pdfBSDF, m_MISmode);
            default: return 0;
            }
        default:
            return 0;
        }
    }

    #define sqr(x) ((x)*(x))
    inline Float mis(Float p1, Float p2, MISMode mode) const {
        switch (m_MISmode)
        {
        case UniformHeuristic: return 0.5;
        case BalanceHeuristic: return p1 / (p1 + p2);
        case PowerHeuristic: return sqr(p1) / (sqr(p1) + sqr(p2));
        default: return 0;
        }
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, Scene* scene, Sampler* sampler, double* cTime, double* cNum) {
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
                Spectrum value = scene->sampleEmitterDirect(dRec, sampler->next2D(),true,cTime);
                *cNum += 1;
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero()) {
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;
                        Float misW = misWeight(bsdfPdf, dRec.pdf, PathNEE);
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
                    Float misW = misWeight(bsdfPdf, lumPdf, PathBSDF);
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