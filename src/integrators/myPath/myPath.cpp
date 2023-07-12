#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

class myPathIntegrator : public SamplingIntegrator {
public:
    MTS_DECLARE_CLASS()
    enum SamplingStrategy {
        PathBSDF = 0,
        PathNEE,
        PathMIS
    };
private:
    int m_maxDepth;
    int m_rrDepth;
    static int m_LiCount;
    std::string m_strategyString;
    SamplingStrategy m_strategy;
    
public:

    /// Initialize the integrator with the specified properties
    myPathIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepth = props.getInteger("maxDepth", 50);
        m_rrDepth = props.getInteger("rrDepth", 0);

        m_strategyString = props.getString("strategy", "mis");
        if (m_strategyString == "bsdf")
            m_strategy = PathBSDF;
        else if (m_strategyString == "nee")
            m_strategy = PathNEE;
        else if (m_strategyString == "mis")
            m_strategy = PathMIS;
        else
            Log(EError, "Unknown strategy: %s", m_strategyString.c_str());
    }

    // Unserialize from a binary data stream
    myPathIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }
    
    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
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
                case PathMIS: return pdfBSDF / (pdfBSDF + pdfDirect);
            }
        case PathNEE:
            switch (m_strategy)
            {
                case PathBSDF: return 0;
                case PathNEE: return 1;
                case PathMIS: return pdfDirect / (pdfBSDF + pdfDirect);
            }
        }
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        ++m_LiCount;
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        if (its.isValid() && its.isEmitter())
            return its.Le(-ray.d);

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid())
                break;
            
            const BSDF *bsdf = its.getBSDF(ray);

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (m_strategy!=PathBSDF && (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero()) {
                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;
                        /* Weight using the power heuristic */
                        Float misW = misWeight(bsdfPdf, dRec.pdf, PathNEE);
                        Li += throughput * value * bsdfVal * misW;
                    }
                }
            }

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;
            
            const Vector wo = its.toWorld(bRec.wo);
            
            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight ;
            eta *= bRec.eta;

            /* Trace a ray in this direction */
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

            if (rRec.depth >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }

            ++rRec.depth;
        }
        return Li;
    }
};

int myPathIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(myPathIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myPathIntegrator, "My path integrator");
MTS_NAMESPACE_END