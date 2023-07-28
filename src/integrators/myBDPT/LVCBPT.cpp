#include <mitsuba/render/scene.h>
#include "myBDPT.h"

#define BDPT_ONLT_PT
#define DEBUG

#undef BDPT_ONLT_PT
#undef DEBUG

MTS_NAMESPACE_BEGIN

class LVCBPTIntegrator : public SamplingIntegrator {
public:
    MTS_DECLARE_CLASS()

private:    
    enum MISMode {
        UniformHeuristic = 0,
        BalanceHeuristic,
        PowerHeuristic
    };
    int m_maxDepthEye;
    int m_maxDepthLight;
    int m_LVCSize;
    Float m_rrEye;
    Float m_rrLight;
    bool m_usePT;
    static int m_LiCount;

    std::string m_MISmodeString;
    MISMode m_MISmode;

public:
    /// Initialize the integrator with the specified properties
    LVCBPTIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_maxDepthLight = props.getInteger("maxDepthLight", 50);
        m_LVCSize = props.getInteger("LVCSize", 50);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_rrLight = props.getFloat("rrLight", 0.6);
        m_usePT = props.getBoolean("usePT", true);

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
    LVCBPTIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }

    void printInfos(){
        std::ostringstream oss;
        oss<<"\n---------- LVCBPT Info Print -----------\n";
        oss<<"maxDepthEye: "<<m_maxDepthEye<<endl;
        oss<<"maxDepthLight: "<<m_maxDepthLight<<endl;
        oss<<"rrEye: "<<m_rrEye<<endl;
        oss<<"rrLight: "<<m_rrLight<<endl;
        oss<<"MISmode: "<<m_MISmodeString<<endl;
        oss<<"usePT: "<<m_usePT<<endl;
        oss<<"BDPTVertex size: "<<sizeof(BDPTVertex)<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        printInfos();
        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        ++m_LiCount;

        Spectrum Li(0.0f);
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

        std::vector<BDPTVertex*> lightPath;
        /* Trace light subpath */
        traceLightSubpath(rRec.its, rRec.sampler, rRec.scene, lightPath);

        /* Connect eye subpath and light subpath */
        connectSubpaths(eyePath, lightPath, rRec.scene, Li);

        /* Clean up */
        for (auto i : eyePath)   delete i;
        for (auto i : lightPath) delete i;

        return Li;
    }
        
    /// wi point outwards
    Float computePdfForward(Vector3& wi, BDPTVertex* mid, BDPTVertex* next) const {
        Intersection its;
        its.p = mid->pos;
        its.shFrame = Frame(mid->n);
        its.uv = mid->uv;
        its.hasUVPartials = 0;
        Vector3 d = next->pos - mid->pos;
        Float distSquared = d.lengthSquared();
        d /= sqrt(distSquared);
        BSDFSamplingRecord bRec(its, its.toLocal(wi), its.toLocal(d), ERadiance);
        Float pdfBsdf = mid->bsdf->pdf(bRec);
        return pdfBsdf * absDot(next->n, d) / distSquared;
    }

    Float computePdfLightDir(BDPTVertex* l, BDPTVertex* e) const {
        PositionSamplingRecord pRec(0.0f);
        pRec.n = l->n;
        Vector3 d = e->pos - l->pos;
        Float distSquared = d.lengthSquared();
        d /= sqrt(distSquared);
        DirectionSamplingRecord dRec(d);
        Float ans = l->e->pdfDirection(dRec, pRec) * absDot(e->n, d) / distSquared;
        return ans;        
    }

int LVCBPTIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(LVCBPTIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(LVCBPTIntegrator, "LVCBPT");
MTS_NAMESPACE_END